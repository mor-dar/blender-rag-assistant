#!/usr/bin/env python3
"""
Blender Documentation Download Script

Downloads Blender documentation in a reproducible, configurable way.
Supports both demo and full dataset modes.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import requests
from bs4 import BeautifulSoup

BLENDER_DOCS_CONFIG = {
    "base_url": "https://docs.blender.org/manual/en/latest/",
    "version": "4.5",
    "last_scraped": datetime.now().strftime("%Y-%m-%d"),
    "sections": {
        "demo": [
            "modeling/meshes/",
            "modeling/curves/",
            "editors/3dview/",
            "scene_layout/",
            "interface/"
        ],
        "full": [
            "modeling/",
            "animation/",
            "rendering/",
            "editors/",
            "getting_started/",
            "compositing/",
            "grease_pencil/",
            "addons/",
            "advanced/",
            "files/"
        ]
    },
    "rate_limit": 1.0,  # seconds between requests
    "max_retries": 3,
    "timeout": 30
}

class BlenderDocsDownloader:
    def __init__(self, config: Dict, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Compatible RAG Assistant Bot)'
        })
        self.downloaded_urls: Set[str] = set()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def download_page(self, url: str, local_path: Path) -> bool:
        """Download a single page with retries"""
        for attempt in range(self.config["max_retries"]):
            try:
                self.logger.info(f"Downloading {url} -> {local_path}")
                response = self.session.get(url, timeout=self.config["timeout"])
                response.raise_for_status()
                
                # Create directory if needed
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save content
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                self.downloaded_urls.add(url)
                time.sleep(self.config["rate_limit"])
                return True
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.config["max_retries"] - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    self.logger.error(f"Failed to download {url} after {self.config['max_retries']} attempts")
                    return False
        return False

    def get_section_urls(self, section_url: str, current_section: str) -> List[str]:
        """Get all URLs in a section by crawling the index"""
        urls = []
        try:
            response = self.session.get(section_url, timeout=self.config["timeout"])
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all internal links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('./') or href.startswith('../'):
                    # Convert relative links to absolute
                    full_url = requests.compat.urljoin(section_url, href)
                elif href.startswith('/manual/'):
                    full_url = f"https://docs.blender.org{href}"
                elif href.startswith('http'):
                    # Skip external links
                    continue
                else:
                    full_url = requests.compat.urljoin(section_url, href)
                
                # Only include HTML pages within the current section being processed
                if (full_url.endswith('.html') and 
                    current_section in full_url and 
                    full_url.startswith(self.config["base_url"])):
                    urls.append(full_url)
        
        except Exception as e:
            self.logger.error(f"Failed to get section URLs for {section_url}: {e}")
        
        return list(set(urls))  # Remove duplicates

    def download_section(self, section: str) -> int:
        """Download all pages in a section"""
        section_url = f"{self.config['base_url']}{section}"
        self.logger.info(f"Processing section: {section}")
        
        # Get main section page
        section_path = self.output_dir / section / "index.html"
        if not self.download_page(section_url, section_path):
            return 0
        
        # Get all URLs in this section
        urls = self.get_section_urls(section_url, section)
        downloaded_count = 1  # Count the index page
        
        for url in urls:
            # Convert URL to local path
            relative_path = url.replace(self.config["base_url"], "")
            local_path = self.output_dir / relative_path
            
            if self.download_page(url, local_path):
                downloaded_count += 1
        
        return downloaded_count

    def save_metadata(self, tier: str, downloaded_count: int):
        """Save metadata about the download"""
        metadata = {
            "config": self.config,
            "tier": tier,
            "downloaded_count": downloaded_count,
            "downloaded_urls": list(self.downloaded_urls),
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = self.output_dir / "download_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved metadata to {metadata_path}")

    def download_tier(self, tier: str) -> int:
        """Download specified tier (demo or full)"""
        if tier not in self.config["sections"]:
            raise ValueError(f"Unknown tier: {tier}. Available: {list(self.config['sections'].keys())}")
        
        sections = self.config["sections"][tier]
        total_downloaded = 0
        
        self.logger.info(f"Starting {tier} tier download of {len(sections)} sections")
        
        for section in sections:
            try:
                count = self.download_section(section)
                total_downloaded += count
                self.logger.info(f"Section {section}: downloaded {count} pages")
            except Exception as e:
                self.logger.error(f"Failed to download section {section}: {e}")
        
        # Save download metadata
        self.save_metadata(tier, total_downloaded)
        
        return total_downloaded

def main():
    parser = argparse.ArgumentParser(description='Download Blender documentation')
    parser.add_argument('--tier', choices=['demo', 'full'], default='demo',
                        help='Which tier to download (default: demo)')
    parser.add_argument('--output-dir', type=Path, 
                        default=Path(__file__).parent.parent / 'data' / 'raw',
                        help='Output directory for downloaded files')
    parser.add_argument('--config', type=Path,
                        help='Custom configuration file (JSON)')
    
    args = parser.parse_args()
    
    # Load custom config if provided
    config = BLENDER_DOCS_CONFIG.copy()
    if args.config and args.config.exists():
        with open(args.config) as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    downloader = BlenderDocsDownloader(config, args.output_dir)
    try:
        total_count = downloader.download_tier(args.tier)
        print(f"Successfully downloaded {total_count} pages for {args.tier} tier")
        print(f"Files saved to: {args.output_dir}")
        
        # Save configuration used
        config_path = args.output_dir / f"config_{args.tier}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"📄 Configuration saved to: {config_path}")
        
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()