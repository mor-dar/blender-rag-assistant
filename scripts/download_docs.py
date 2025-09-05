#!/usr/bin/env python3
"""
Blender Documentation Download Script

Downloads Blender documentation in HTML or EPUB format.
Supports both demo and full dataset modes, plus direct archive downloads.
"""

import argparse
import json
import logging
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import requests
from bs4 import BeautifulSoup

# Add path to import project config
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.config import BLENDER_VERSION

BLENDER_DOCS_CONFIG = {
    "base_url": f"https://docs.blender.org/manual/en/{BLENDER_VERSION}/",
    "version": BLENDER_VERSION,
    "last_scraped": datetime.now().strftime("%Y-%m-%d"),
    "archive_urls": {
        "html": f"https://docs.blender.org/manual/en/{BLENDER_VERSION}/blender_manual_html.zip",
        "epub": f"https://docs.blender.org/manual/en/{BLENDER_VERSION}/blender_manual_epub.zip"
    },
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

    def download_archive(self, format_type: str) -> bool:
        """Download and extract HTML or EPUB archive"""
        if format_type not in self.config["archive_urls"]:
            self.logger.error(f"Unknown format: {format_type}. Available: {list(self.config['archive_urls'].keys())}")
            return False
        
        url = self.config["archive_urls"][format_type]
        archive_path = self.output_dir / f"blender_manual_{format_type}.zip"
        
        try:
            self.logger.info(f"Downloading {format_type} archive from {url}")
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            # Download with progress
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Archive downloaded to {archive_path}")
            
            # Extract archive with flattened structure
            extract_dir = self.output_dir / f"blender_manual_{format_type}"
            extract_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Extracting archive to {extract_dir}")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Get list of all files in the archive
                file_list = zip_ref.namelist()
                
                # Find the root directory in the archive (usually something like blender_manual_v450_en.html/)
                root_dirs = set()
                for file_path in file_list:
                    if '/' in file_path:
                        root_dir = file_path.split('/')[0]
                        root_dirs.add(root_dir)
                
                # If there's a single root directory, we'll flatten by extracting its contents directly
                if len(root_dirs) == 1:
                    root_dir = list(root_dirs)[0]
                    self.logger.info(f"Flattening archive structure, removing root directory: {root_dir}")
                    
                    for file_info in zip_ref.infolist():
                        if file_info.filename.startswith(root_dir + '/') and not file_info.is_dir():
                            # Remove the root directory from the path
                            relative_path = file_info.filename[len(root_dir) + 1:]
                            if relative_path:  # Skip empty paths
                                target_path = extract_dir / relative_path
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                
                                # Extract the file
                                with zip_ref.open(file_info) as source:
                                    with open(target_path, 'wb') as target:
                                        target.write(source.read())
                else:
                    # Fallback to normal extraction if structure is unexpected
                    zip_ref.extractall(extract_dir)
            
            # Clean up zip file
            archive_path.unlink()
            
            self.logger.info(f"Successfully extracted {format_type} manual to {extract_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download/extract {format_type} archive: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Download Blender documentation')
    parser.add_argument('--format', choices=['html', 'epub'], default='html',
                        help='Download format: html (archive) or epub (archive)')
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
    
    # Download archive format
    downloader = BlenderDocsDownloader(config, args.output_dir)
    try:
        success = downloader.download_archive(args.format)
        if success:
            print(f"Successfully downloaded {args.format.upper()} manual")
            print(f"Files saved to: {args.output_dir / f'blender_manual_{args.format}'}")
        else:
            print(f"Failed to download {args.format.upper()} manual")
            sys.exit(1)
        
        # Save configuration used
        config_path = args.output_dir / f"config_{args.format}_archive.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to: {config_path}")
        
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()