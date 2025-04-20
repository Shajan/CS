import asyncio
import os
from typing import List, Dict
from urllib.parse import urlparse
from functools import lru_cache
import time
import random

@lru_cache(maxsize=1000)
def host_name_slow(url: str) -> str:
    return urlparse(url).hostname.lower()

@lru_cache(maxsize=1000)
def host_name(url: str) -> str:
    start = url.find("://")
    if start != -1:
        start += 3
    else:
        start = 0
    end = url.find("/", start)
    if end == -1:
        end = len(url)
    return url[start:end].lower()


class HtmlParser:
    def __init__(self, web_graph: Dict[str, List[str]]):
        self.web_graph = web_graph

    def getUrls(self, url: str) -> List[str]:
        # Simulate network latency
        time.sleep(random.uniform(0.001, 0.005))
        return self.web_graph.get(url, [])


class Solution:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        return asyncio.run(self.async_crawl(startUrl, htmlParser))

    async def async_crawl(self, startUrl, htmlParser: 'HtmlParser') -> List[str]:
        num_workers = os.cpu_count() or 1
        num_parallel_fetches = 10
        num_parallel = asyncio.Semaphore(num_parallel_fetches)
        #lock = asyncio.Lock()
        allowed_host = host_name(startUrl)
        task_queue = asyncio.Queue()
        await task_queue.put(startUrl)
        links = set([startUrl])

        async def worker(task_queue: asyncio.Queue):
            while True:
                url = await task_queue.get()
                try:
                    async with num_parallel:
                        urls = await asyncio.to_thread(htmlParser.getUrls, url)
                    for candidate in urls:
                        if host_name(candidate) != allowed_host:
                            continue
                        #add = False
                        #async with lock:
                        if candidate not in links:
                            links.add(candidate)
                            #add = True
                        #if add:
                            await task_queue.put(candidate)
                except Exception as e:
                    print(f"Error fetching {url}: {e}")
                finally:
                    task_queue.task_done()

        workers = [asyncio.create_task(worker(task_queue)) for _ in range(num_workers)]
        await task_queue.join()

        for worker in workers:
            worker.cancel()

        return list(links)


# Utility to generate a large mock web graph
def generate_large_web_graph(domain: str, total_pages: int, max_links_per_page: int = 5) -> Dict[str, List[str]]:
    base = f"https://{domain}"
    urls = [f"{base}/page{i}" for i in range(total_pages)]
    
    # Introduce 25% of the URLs as frequently reused "popular" links
    num_shared_urls = total_pages // 4
    shared_urls = random.sample(urls, num_shared_urls)

    graph = {}
    for url in urls:
        links = set()
        # Add 1–2 popular shared URLs to most pages
        links.update(random.sample(shared_urls, k=random.randint(1, 2)))
        # Add some unique or less common links to mix things up
        links.update(random.sample(urls, k=random.randint(0, max_links_per_page - len(links))))
        graph[url] = list(links)
    
    return graph


# Create test and profile
if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)

    total_pages = 1000
    web_graph = generate_large_web_graph("foo.bar", total_pages)
    parser = HtmlParser(web_graph)

    start_url = "https://foo.bar/page0"
    solution = Solution()

    start_time = time.time()
    crawled = solution.crawl(start_url, parser)
    elapsed = time.time() - start_time

    print(f"\nCrawled {len(crawled)} pages out of {total_pages} in {elapsed:.2f} seconds.")
    print(f"host_name cache info: {host_name.cache_info()}")

