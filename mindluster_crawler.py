"""
MindLuster course crawler
==========================

Walks the site 4 levels deep and produces one CSV per category of every
lesson video it finds:

    Level 1  /free-online-courses            -> list of categories
    Level 2  /certified/cat/{id}/{slug}       -> list of courses
    Level 3  /certificate/{id}/{slug}-video   -> list of lessons
    Level 4  /lesson/{id}-video               -> the actual video (YouTube embed)

Robustness features:
    - Retry with backoff: a stuck page gets up to 4 attempts (5s, 10s, 15s
      growing waits) before the script gives up on that page.
    - Per-course checkpointing: every course's lessons are written to
      mindluster_data/<category>.csv immediately after that course finishes,
      and mindluster_progress.json tracks the last completed course id per
      category (not just once a whole category finishes).
    - Auto-resume: if a category still gets stuck after retries, it's logged
      and the script moves to the next category. Just re-run the script
      (python mindluster_crawler.py) -- no course number to type in. It reads
      the progress file, reopens that category, skips (no re-scraping) the
      courses it already has, and continues from exactly where it stopped.
    - Jittered delay: random jitter is added to the delay between requests so
      the request pattern isn't a perfectly metronomic, easily-throttled beat.

Install deps:
    pip install requests beautifulsoup4

Run:
    python mindluster_crawler.py                            # crawl everything, auto-resumes if re-run
    python mindluster_crawler.py --category-id 19            # just one category (e.g. Mathematics)
    python mindluster_crawler.py --max-categories 2 --max-courses 5 --max-lessons 3   # quick test
    python mindluster_crawler.py --fresh                     # ignore saved progress, start over

Output (all under ./mindluster_data/):
    mindluster_data/<category_name>.csv   one file per category, appended to incrementally
    mindluster_data/progress.json         per-category list of completed course ids + status
"""

import argparse
import csv
import json
import re
import time
import random
import logging
import os
from dataclasses import dataclass, asdict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.mindluster.com"
DATA_DIR = "mindluster_data"
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mindluster")

SOURCE_NAME = "MindLuster"   # all rows come from this one site, so courseSource is constant

RETRY_WAITS = [5, 10, 15]  # seconds -- one wait per retry after a failed attempt


@dataclass
class Lesson:
    category_id: str
    category_name: str
    course_id: str
    course_title: str
    course_url: str
    lesson_id: str
    lesson_name: str
    lesson_url: str
    course_duration: str    # holds this lesson's duration (e.g. "12:34")
    course_source: str      # constant "MindLuster" - kept for schema parity
                             # with other crawlers that scrape multiple providers
    # --- extra fields beyond the requested schema, kept because they're
    #     real scraped data (the whole point of the level-4 crawl) - they
    #     just tack on after the 10 normalized columns, not in place of them
    video_id: str          # YouTube video id, if found
    video_embed_url: str   # full embed url, if found


# Maps internal field name -> the exact column name you specified.
# csv fieldnames / row dicts are built from this, in this order.
COLUMN_RENAME = {
    "category_id": "categoryId",
    "category_name": "categoryName",
    "course_id": "courseId",
    "course_title": "courseTitle",
    "course_url": "courseUrl",
    "lesson_id": "lessonId",
    "lesson_name": "lessonName",
    "lesson_url": "lessonUrl",
    "course_duration": "courseDuration",
    "course_source": "courseSource",
    "video_id": "videoId",
    "video_embed_url": "videoEmbedUrl",
}


def to_row_dict(lesson: "Lesson") -> dict:
    """Lesson -> a dict keyed by the normalized column names, for csv writing."""
    raw = asdict(lesson)
    return {COLUMN_RENAME[k]: v for k, v in raw.items()}


def slugify(name: str) -> str:
    """Turn a category/course name into the site's URL-slug convention (spaces -> hyphens)."""
    slug = re.sub(r"\s+", "-", name.strip())
    slug = re.sub(r"[^\w\-]", "", slug)
    return slug


def extract_name_before_count(text: str, label: str) -> str:
    """
    MindLuster link text repeats the name on both sides of the count, e.g.
    'Mathematics 308 course Mathematics' or
    'Python programming language 15 Lessons Python programming language'.
    Take everything before the number so the name isn't duplicated.
    Falls back to the raw text if the pattern isn't found.
    """
    m = re.match(rf"^(.*?)\s+\d+\s*{re.escape(label)}\b", text, re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return text.strip()


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "unnamed"


class ProgressTracker:
    """Tracks, per category, which course ids are already fully scraped."""

    def __init__(self, path=PROGRESS_FILE):
        self.path = path
        self.data = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                log.info("Loaded progress file with %d categor%s already tracked",
                          len(self.data), "y" if len(self.data) == 1 else "ies")
            except (json.JSONDecodeError, OSError):
                log.warning("Progress file was unreadable, starting fresh")
                self.data = {}

    def get_completed_courses(self, category_id):
        return set(self.data.get(category_id, {}).get("completed_course_ids", []))

    def mark_course_done(self, category_id, category_name, course_id):
        entry = self.data.setdefault(category_id, {
            "category_name": category_name,
            "completed_course_ids": [],
            "status": "in_progress",
        })
        if course_id not in entry["completed_course_ids"]:
            entry["completed_course_ids"].append(course_id)
        self._save()

    def mark_category_done(self, category_id):
        if category_id in self.data:
            self.data[category_id]["status"] = "done"
        self._save()

    def mark_category_stuck(self, category_id, reason):
        entry = self.data.setdefault(category_id, {
            "category_name": "", "completed_course_ids": [], "status": "in_progress",
        })
        entry["status"] = f"stuck: {reason}"
        self._save()

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)


class MindlusterCrawler:
    def __init__(self, delay=1.0, retries=4, timeout=20):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay
        self.retries = retries
        self.timeout = timeout

    # ---------- low level fetch with retry / backoff / jitter ----------
    def get(self, url):
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    self._jittered_sleep()
                    return resp.text
                log.warning("GET %s -> HTTP %s (attempt %d/%d)", url, resp.status_code, attempt, self.retries)
            except requests.RequestException as e:
                log.warning("GET %s failed: %s (attempt %d/%d)", url, e, attempt, self.retries)

            if attempt < self.retries:
                wait = RETRY_WAITS[min(attempt - 1, len(RETRY_WAITS) - 1)]
                log.info("  retrying in %ds...", wait)
                time.sleep(wait)

        log.error("Giving up on %s after %d attempts", url, self.retries)
        return None

    def _jittered_sleep(self):
        # +/- 30% jitter around the base delay so requests aren't metronomic
        jittered = self.delay * random.uniform(0.7, 1.3)
        time.sleep(jittered)

    # ---------- Level 1: categories ----------
    def get_categories(self):
        html = self.get(f"{BASE}/free-online-courses")
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        categories = []
        for a in soup.select('a[href*="/certified/cat/"]'):
            href = a.get("href", "")
            m = re.search(r"/certified/cat/(\d+)", href)
            if not m:
                continue
            cat_id = m.group(1)
            name = extract_name_before_count(a.get_text(strip=True), "course")
            categories.append({"id": cat_id, "name": name, "url": urljoin(BASE, href)})
        seen = {}
        for c in categories:
            seen[c["id"]] = c
        return list(seen.values())

    # ---------- Level 2: courses within a category (handles pagination) ----------
    def get_courses_for_category(self, category_id, category_name):
        """
        The category landing page (/certified/cat/{id}/{slug}) only server-renders
        page 1; the rest loads via infinite-scroll AJAX (POST to /load_all_courses/{id})
        that needs a live CSRF token + session cookie. Instead we use the site's
        parallel, stateless, paginated endpoint: /courses/cat/{id}/{slug}?page=N,
        which has real Previous/Next <a> links -- no auth or session state required.
        """
        slug = slugify(category_name)
        start_url = f"{BASE}/courses/cat/{category_id}/{slug}?page=1"

        courses = self._paginate_courses(start_url)

        # fallback: if the slug guess didn't resolve to anything, try without a slug
        if not courses:
            fallback_url = f"{BASE}/courses/cat/{category_id}/?page=1"
            log.info("  slug '%s' returned nothing, retrying without slug", slug)
            courses = self._paginate_courses(fallback_url)

        return courses

    def _paginate_courses(self, start_url):
        courses = []
        seen_ids = set()
        url = start_url
        page_num = 1
        while True:
            html = self.get(url)
            if not html:
                break
            soup = BeautifulSoup(html, "html.parser")
            found_this_page = 0
            for a in soup.select('a[href*="/certificate/"]'):
                href = a.get("href", "")
                m = re.search(r'/certificate/(\d+)/([^/"]+)', href)
                if not m:
                    continue
                course_id = m.group(1)
                if course_id in seen_ids:
                    continue
                seen_ids.add(course_id)
                found_this_page += 1
                text = a.get_text(" ", strip=True)
                lessons_match = re.search(r"(\d+)\s*Lessons", text)
                name = a.get("title") or extract_name_before_count(text, "Lessons")
                courses.append({
                    "id": course_id,
                    "name": name,
                    "url": urljoin(BASE, href),
                    "lesson_count": lessons_match.group(1) if lessons_match else "",
                })

            log.info("    page %d -> %d new courses (running total %d)", page_num, found_this_page, len(courses))

            # find the real "Next" link (its visible text is literally "Next")
            next_link = None
            for a in soup.find_all("a"):
                if a.get_text(strip=True).lower() == "next" and a.get("href"):
                    next_link = a
                    break

            if not next_link:
                break  # last page reached

            candidate = urljoin(BASE, next_link["href"])
            if candidate == url:
                break  # safety: don't loop forever on a self-referential link
            url = candidate
            page_num += 1
            if page_num > 2000:  # sanity valve, no category should have this many pages
                log.warning("    hit 2000-page safety cap, stopping")
                break

        return courses

    # ---------- Level 3: lessons within a course ----------
    def get_lessons_for_course(self, course_url):
        html = self.get(course_url)
        if not html:
            return None  # distinguish "failed" from "genuinely zero lessons"
        soup = BeautifulSoup(html, "html.parser")
        lessons = []
        seen_ids = set()
        for a in soup.select('a[href*="/lesson/"]'):
            href = a.get("href", "")
            m = re.search(r"/lesson/(\d+)-video", href)
            if not m:
                continue
            lesson_id = m.group(1)
            if lesson_id in seen_ids:
                continue
            seen_ids.add(lesson_id)
            text = a.get_text(" ", strip=True)
            dur_match = re.search(r"(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})", text)
            duration = dur_match.group(1) if dur_match else ""
            title = a.get("title") or text
            title = re.sub(r"\d{1,2}:\d{2}(:\d{2})?", "", title).strip()
            lessons.append({
                "id": lesson_id,
                "name": title,
                "url": urljoin(BASE, href),
                "duration": duration,
            })
        return lessons

    # ---------- Level 4: the actual video on a lesson page ----------
    def get_video_for_lesson(self, lesson_url):
        html = self.get(lesson_url)
        if not html:
            return {"video_id": "", "video_embed_url": ""}
        soup = BeautifulSoup(html, "html.parser")

        iframe = soup.find("iframe", src=re.compile(r"youtube\.com/embed/"))
        if iframe and iframe.get("src"):
            vid = re.search(r"youtube\.com/embed/([A-Za-z0-9_-]{6,})", iframe["src"])
            if vid:
                return {"video_id": vid.group(1), "video_embed_url": iframe["src"]}

        iframe2 = soup.find("iframe", attrs={"data-src": re.compile(r"youtube\.com/embed/")})
        if iframe2:
            vid = re.search(r"youtube\.com/embed/([A-Za-z0-9_-]{6,})", iframe2["data-src"])
            if vid:
                return {"video_id": vid.group(1), "video_embed_url": iframe2["data-src"]}

        m = re.search(r"(?:youtube\.com/(?:embed/|watch\?v=)|youtu\.be/)([A-Za-z0-9_-]{6,})", html)
        if m:
            vid_id = m.group(1)
            return {"video_id": vid_id, "video_embed_url": f"https://www.youtube.com/embed/{vid_id}"}

        return {"video_id": "", "video_embed_url": ""}

    # ---------- per-category CSV writer (append mode, checkpointed) ----------
    def _open_category_csv(self, category_name):
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, f"{safe_filename(category_name)}.csv")
        file_exists = os.path.exists(path)
        f = open(path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=list(COLUMN_RENAME.values()))
        if not file_exists:
            writer.writeheader()
        return f, writer, path

    # ---------- orchestration ----------
    def crawl(self, category_id=None, max_categories=None, max_courses=None, max_lessons=None,
              fetch_videos=True, fresh=False):

        if fresh and os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            log.info("--fresh: cleared saved progress")

        tracker = ProgressTracker()

        categories = self.get_categories()
        if category_id:
            categories = [c for c in categories if c["id"] == str(category_id)]
        if max_categories:
            categories = categories[:max_categories]

        log.info("Found %d categor%s to crawl", len(categories), "y" if len(categories) == 1 else "ies")

        for ci, cat in enumerate(categories, 1):
            if tracker.data.get(cat["id"], {}).get("status") == "done":
                log.info("[%d/%d] Category '%s' already marked done, skipping", ci, len(categories), cat["name"])
                continue

            log.info("[%d/%d] Category: %s (id=%s)", ci, len(categories), cat["name"], cat["id"])
            already_done_courses = tracker.get_completed_courses(cat["id"])
            if already_done_courses:
                log.info("  resuming: %d courses already completed in this category", len(already_done_courses))

            courses = self.get_courses_for_category(cat["id"], cat["name"])
            if not courses:
                log.error("  could not load any courses for '%s' (site unreachable or genuinely empty), "
                          "marking stuck and moving on", cat["name"])
                tracker.mark_category_stuck(cat["id"], "course list unreachable or empty")
                continue
            if max_courses:
                courses = courses[:max_courses]

            log.info("  -> %d courses total", len(courses))

            csv_file, writer, csv_path = self._open_category_csv(cat["name"])
            try:
                for coi, course in enumerate(courses, 1):
                    if course["id"] in already_done_courses:
                        continue  # fast-forward, no re-scraping

                    log.info("  [%d/%d] Course: %s (id=%s)", coi, len(courses), course["name"], course["id"])
                    lessons = self.get_lessons_for_course(course["url"])
                    if lessons is None:
                        log.warning("    could not load lessons for course '%s', skipping this course "
                                    "(will retry on next run)", course["name"])
                        continue
                    if max_lessons:
                        lessons = lessons[:max_lessons]
                    log.info("    -> %d lessons", len(lessons))

                    course_rows = []
                    course_ok = True
                    for lesson in lessons:
                        video_info = {"video_id": "", "video_embed_url": ""}
                        if fetch_videos:
                            video_info = self.get_video_for_lesson(lesson["url"])

                        row = Lesson(
                            category_id=cat["id"],
                            category_name=cat["name"],
                            course_id=course["id"],
                            course_title=course["name"],
                            course_url=course["url"],
                            lesson_id=lesson["id"],
                            lesson_name=lesson["name"],
                            lesson_url=lesson["url"],
                            course_duration=lesson["duration"],
                            course_source=SOURCE_NAME,
                            video_id=video_info["video_id"],
                            video_embed_url=video_info["video_embed_url"],
                        )
                        course_rows.append(row)

                    # write + checkpoint only once the whole course succeeded
                    for row in course_rows:
                        writer.writerow(to_row_dict(row))
                    csv_file.flush()
                    tracker.mark_course_done(cat["id"], cat["name"], course["id"])

                tracker.mark_category_done(cat["id"])
                log.info("  category '%s' complete -> %s", cat["name"], csv_path)
            finally:
                csv_file.close()

        log.info("Done. Data in ./%s/, progress in %s", DATA_DIR, PROGRESS_FILE)


def main():
    parser = argparse.ArgumentParser(description="Crawl mindluster.com courses -> lessons -> videos")
    parser.add_argument("--category-id", type=str, default=None, help="Only crawl this category id (e.g. 19 for Mathematics)")
    parser.add_argument("--max-categories", type=int, default=None)
    parser.add_argument("--max-courses", type=int, default=None, help="Max courses per category")
    parser.add_argument("--max-lessons", type=int, default=None, help="Max lessons per course")
    parser.add_argument("--no-videos", action="store_true", help="Skip level-4 video fetch (faster, just lists lessons)")
    parser.add_argument("--delay", type=float, default=1.0, help="Base delay between requests (seconds); jitter is added automatically")
    parser.add_argument("--fresh", action="store_true", help="Ignore any saved progress and start over")
    args = parser.parse_args()

    crawler = MindlusterCrawler(delay=args.delay)
    crawler.crawl(
        category_id=args.category_id,
        max_categories=args.max_categories,
        max_courses=args.max_courses,
        max_lessons=args.max_lessons,
        fetch_videos=not args.no_videos,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
