import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

class JobSearchScraper:

    def __init__(self):
        self.api_key = os.getenv("JSEARCH_API_KEY")
        self.base_url = "https://jsearch.p.rapidapi.com/search"

        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }

    def search_jobs(self, query, location, num_pages=5):

        jobs = []

        for page in range(num_pages):

            params = {
                "query": query,
                "location": location,
                "page": page + 1
            }

            try:
                response = requests.get(
                    self.base_url,
                    headers=self.headers,
                    params=params
                )

                response.raise_for_status()

                data = response.json()

                if "data" in data:

                    for job in data["data"]:

                        jobs.append({
                            "title": job.get("job_title"),
                            "company": job.get("employer_name"),
                            "location": job.get("job_city"),
                            "salary_min": job.get("job_min_salary"),
                            "salary_max": job.get("job_max_salary"),
                            "description": job.get("job_description"),
                            "job_type": job.get("job_employment_type"),
                            "posted_date": job.get("job_posted_at_datetime_utc"),
                            "url": job.get("job_apply_link")
                        })

                print(f"✓ Scraped page {page+1}")

            except requests.exceptions.RequestException as e:

                print(f"✗ Error on page {page+1}: {e}")

        return pd.DataFrame(jobs)

    def save_to_csv(self, df, filename="data/raw/jobs.csv"):

        df.to_csv(filename, index=False)

        print(f"✓ Data saved to {filename}")
        print(f"Total jobs: {len(df)}")


if __name__ == "__main__":

    scraper = JobSearchScraper()

    print("Searching jobs with JSearch API...")

    jobs_df = scraper.search_jobs(
        query="Data Scientist",
        location="San Francisco",
        num_pages=5
    )

    scraper.save_to_csv(jobs_df)

    print(jobs_df.head())