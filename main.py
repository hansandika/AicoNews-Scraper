from dotenv import load_dotenv
import json
from typing import List
import os
from datetime import datetime
from urllib.parse import urlencode
import requests
import uuid
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urljoin
import re 
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dataclasses import dataclass
import psycopg2
from psycopg2 import Error
import time
from slugify import slugify
import html
@dataclass
class News:
    headline: str
    slug: str 
    content: str
    content_html: str
    source: str
    source_url: str
    image_url: str
    category: str 
    author: str
    published_date: datetime

def remove_empty_tags(parent_tag):
    if parent_tag.attrs and 'class' in parent_tag.attrs:
        del parent_tag['class']

    empty_tags = parent_tag.find_all(lambda tag: tag.name != 'br' and not tag.contents and (not tag.string or not tag.string.strip()))

    for empty_tag in empty_tags:
        empty_tag.extract()

def remove_unwanted_attributes(tag):
    if tag is not None and not isinstance(tag, NavigableString):
        # Remove all attributes except href
        if tag.attrs:
            for attribute in list(tag.attrs):
                if attribute not in ['href']:
                    del tag[attribute]
        
        # Remove All attributes from child tags
        for child in tag.find_all():
            if child.attrs:
                for attribute in list(child.attrs):
                    if attribute not in ['href']:
                        del child[attribute]

def remove_whitespace(tag):
    if tag is not None:
        if tag.string:
            cleaned_text = tag.string.strip()
            tag.string.replace_with(cleaned_text)
        elif tag.children:
            children = list(tag.children)
            sizeOfChildren = len(children)
            index = 0
            for child in tag.children:  
                if child.string and index == 0:
                    cleaned_text = child.string.lstrip()
                    child.string.replace_with(cleaned_text)
                elif child.string and index == sizeOfChildren - 1:
                    cleaned_text = child.string.rstrip()
                    child.string.replace_with(cleaned_text)
                index += 1

def get_scrapeops_url(url):
    SCRAPE_OPS_API_KEY = os.getenv('SCRAPE_OPS_API_KEY')
    payload = {'api_key': SCRAPE_OPS_API_KEY, 'url': url}
    proxy_url = 'https://proxy.scrapeops.io/v1/?' + urlencode(payload)
    return proxy_url

def request_page(url):
    # Send a GET request to the URL
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.84 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    }
    response = requests.get(url=get_scrapeops_url(url),headers=headers)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None

def get_existing_slugs_from_postgres():
    try:
        print("Connecting to PostgreSQL to retrieve existing slugs...")
        connection = psycopg2.connect(
            user=os.getenv('DB_USERNAME'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME') if os.getenv('ENVIRONMENT') == 'local' else os.getenv('DB_NAME_PROD')
        )
        
        cursor = connection.cursor()
        cursor.execute("SELECT slug FROM news")
        existing_slugs = {row[0] for row in cursor.fetchall()}
        return existing_slugs

    except (Exception, Error) as error:
        print("Error while retrieving data from PostgreSQL:", error)
        return set()

    finally:
        if connection:
            cursor.close()
            connection.close()

def store_news_in_postgres(news: List[News]):
    try:
        print("Connecting to PostgreSQL...")
        startTime = time.time()
        connection = psycopg2.connect(
            user=os.getenv('DB_USERNAME'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME') if os.getenv('ENVIRONMENT') == 'local' else os.getenv('DB_NAME_PROD')
        )
        
        cursor = connection.cursor()

        existing_slugs = get_existing_slugs_from_postgres()

        news = [news_obj for news_obj in news if news_obj.slug not in existing_slugs]

        insert_query = """INSERT INTO news (headline, slug, content, content_html, source, source_url, thumbnail_url, author_name, category_name,  published_date) 
                          VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        
        records_to_insert = [(news_obj.headline, news_obj.slug, news_obj.content, news_obj.content_html, news_obj.source, news_obj.source_url, news_obj.image_url, news_obj.author, news_obj.category, news_obj.published_date.strftime("%Y-%m-%d %H:%M:%S")) for news_obj in news]

        cursor.executemany(insert_query, records_to_insert)
        connection.commit()
        print("Records inserted successfully into news table")

        endTime = time.time()
        print(f"Time taken to insert records: {endTime - startTime} seconds")

    except (Exception, Error) as error:
        print("Error while inserting data into PostgreSQL:", error)

    finally:
        if connection:
            cursor.close()
            connection.close()

def scrape_from_investing_website(url, existing_urls):
    soup = request_page(url)

    news: List[News] = list()
    unique_links = set()

    if soup == None:
        return news

    links = soup.find_all('a', attrs={'data-test': 'article-title-link'})

    if not links:
        return news
        
    for link in links:
        if 'href' in link.attrs and len(unique_links) < 5:
            href_value = link['href']
            full_link = urljoin(url, href_value)  # Convert relative link to absolute link
            if full_link not in existing_urls:
                unique_links.add(full_link)

    # Scraping content of each linked page
    for linked_page in unique_links:
        print(f"\nScraping content from: {linked_page}")

        linked_page_soup = request_page(linked_page)

        if linked_page_soup == None:
            print('Linked Page Not Found')
            continue
        
        # Find the script tag with type="application/ld+json"
        script_tag = linked_page_soup.find('script', type='application/ld+json')

        if not script_tag:
            print("Script Tag Not Found")
            continue
        
        # Extract the content of the script tag
        script_content = script_tag.string.strip()

        # Parse the script content
        script_data = json.loads(script_content)

        headline = script_data.get('headline')
        date_published_str = script_data.get('datePublished')
        if date_published_str == None:
            print('Date Published Not Found')
            continue
        
        date_published = datetime.strptime(date_published_str, '%Y-%m-%dT%H:%M:%S.%f%z') 
        
        author_name = script_data.get('author').get('name') 
        thumbnail_url = script_data.get('image').get('url')
        category= script_data.get('articleSection')

        print('Headline: ', headline)
        print("Date Published: ", date_published)
        print("Author Name: ", author_name)
        print("Thumbnail URL: ", thumbnail_url)

        articlePageContent = linked_page_soup.find('div', {'class': 'article_articlePage__UMz3q'})

        if not articlePageContent:
            print("Article content Not Found")
            continue 

        elements_with_tags = articlePageContent.find_all(lambda tag: tag.name == 'p' and (tag.find_parent('div') and 'article_articlePage__UMz3q' in tag.find_parent('div').get('class', [])))

        if not elements_with_tags:
            print("Article content Not Found")
            continue

        # Fix href url
        for paragraph in elements_with_tags:
            # Find all <a> tags within the paragraph
            links = paragraph.find_all('a')
            # Iterate through each <a> tag
            for link in links:
                # Check if href attribute is present and starts with '/'
                if 'href' in link.attrs and link['href'].startswith('/'):
                    # Modify the href attribute to include the full path
                    link['href'] = urljoin('https://www.investing.com',link['href'])
        
        # Apply the remove_unwanted_attributes function to all tags
        for element_with_tag in elements_with_tags:
            if element_with_tag is not None:
                remove_unwanted_attributes(element_with_tag)
                remove_empty_tags(element_with_tag)
                remove_whitespace(element_with_tag)

        news_content_tag = '\n'.join([str(paragraph) for paragraph in elements_with_tags if paragraph.get_text(strip=True).strip() != ''])

        print('Content with tag:', news_content_tag)
        
        # Remove all tags from the soup
        news_content = '\n'.join([paragraph.get_text(strip=True,separator=' ') for paragraph in elements_with_tags if paragraph.get_text(strip=True).strip() != ''])

        print("\nContent without tag:", news_content)
        print("\n\n\n")

        news.append(News(
            headline=headline,
            slug=slugify(headline),
            content=news_content,
            content_html=html.escape(news_content_tag),
            source='Investing',
            source_url=linked_page,
            image_url=thumbnail_url,
            category=category.capitalize(),
            author=author_name,
            published_date=date_published
        ))

    return news

def scrape_from_cnbc_website(url, existing_urls):
    soup = request_page(url)

    news: List[News] = list()
    unique_links = set()

    if soup == None:
        return news

    target_div = soup.find('div', {'id': 'SectionWithNativeTVE-ThreeUpStack-6'})
    if not target_div:
        print('Initial Content for title and href not Found')
        return news

    pattern = re.compile(r'https://www\.cnbc\.com/\d{4}/\d{2}/\d{2}')

    links = target_div.find_all('a', href=lambda href: href and pattern.match(href))

    for link in links:
        if 'href' in link.attrs and len(unique_links) < 5:
            href_value = link['href']
            full_link = urljoin(url, href_value)  # Convert relative link to absolute link
            if full_link not in existing_urls:
                unique_links.add(full_link)

    # Scraping content of each linked page
    for linked_page in unique_links:
        print(f"\nScraping content from: {linked_page}")

        linked_page_soup = request_page(linked_page)

        if linked_page_soup == None:
            continue

        # Find the script tag with type="application/ld+json"
        script_tag = linked_page_soup.find('script', type='application/ld+json')

        if not script_tag:
            print("Script Tag Not Found")
            continue
        
        # Extract the content of the script tag
        script_content = script_tag.string.strip()

        # Parse the script content
        script_data = json.loads(script_content)

        headline = script_data.get('headline')
        date_published_str = script_data.get('datePublished') 
        if date_published_str == None:
            print('Date Published Not Found')
            continue
        
        date_published = datetime.strptime(date_published_str,'%Y-%m-%dT%H:%M:%S%z')

        author= script_data.get('author')
        author_name = ''
        if len(author) != 0:
            author_name= author[0].get('name')
        thumbnail_url = script_data.get('thumbnailUrl')
        category= script_data.get('articleSection')

        print('Headline: ', headline)
        print("Date Published: ", date_published)
        print("Author Name: ", author_name)
        print("Thumbnail URL: ", thumbnail_url)

        articlePageContent = linked_page_soup.find('div', {'class': 'ArticleBody-articleBody'})

        if not articlePageContent:
            print("Article content Not Found")
            continue 
        
        elements_with_tags = articlePageContent.find_all(lambda tag: (tag.name == 'p' and (tag.find_parent('div') and 'group' in tag.find_parent('div').get('class', []))) or (tag.name == 'h2' and 'ArticleBody-subtitle' in tag.get('class',[])))
        
        if not elements_with_tags:
            print("Article content Not Found")
            continue

        # Fix href url
        for paragraph in elements_with_tags:
            # Find all <a> tags within the paragraph
            links = paragraph.find_all('a')
            # Iterate through each <a> tag
            for link in links:
                # Check if href attribute is present and starts with '/'
                if 'href' in link.attrs and link['href'].startswith('/'):
                    # Modify the href attribute to include the full path
                    link['href'] = urljoin('https://www.cnbc.com/',link['href'])
        
        # Apply the remove_unwanted_attributes function to all tags 
        for element_with_tag in elements_with_tags:
            if element_with_tag is not None:
                remove_unwanted_attributes(element_with_tag)
                remove_empty_tags(element_with_tag)
                remove_whitespace(element_with_tag)

        news_content_tag = '\n '.join([str(paragraph) for paragraph in elements_with_tags if paragraph.get_text(strip=True).strip() != '' and 'CNBC PRO' not in paragraph.get_text(strip=True).strip()])

        print('Content with tag:', news_content_tag)
        
        # Remove all tags from the soup
        news_content = '\n'.join([paragraph.get_text(strip=True,separator=' ') for paragraph in elements_with_tags if paragraph.get_text(strip=True).strip() != '' and 'CNBC PRO' not in paragraph.get_text(strip=True).strip()])

        print("\nContent without tag:", news_content)
        print("\n\n\n")
        
        news.append(News(
            headline=headline,
            slug=slugify(headline),
            content=news_content,
            content_html=html.escape(news_content_tag),
            source='CNBC',
            source_url=linked_page,
            image_url=thumbnail_url,
            category=category.capitalize(),
            author=author_name,
            published_date=date_published,
        ))

    return news

def scrape_from_cnn_website(url, existing_urls):
    soup = request_page(url)

    news: List[News] = list()
    unique_links = set()

    if soup == None:
        return news

    
    target_div = soup.find_all('div', {'data-open-link': lambda x: x and x.endswith("/index.html")})

    if not target_div:
        print('Initial Content for title and href not Found')
        return news
    
    links = [div['data-open-link'] for div in target_div]
    
    for link in links:
        if len(unique_links) < 5:
            full_link = urljoin('https://edition.cnn.com', link)  # Convert relative link to absolute link
            if full_link not in existing_urls:
                unique_links.add(full_link)

    # Scraping content of each linked page
    for linked_page in unique_links:
        print(f"\nScraping content from: {linked_page}")

        linked_page_soup = request_page(linked_page)

        if linked_page_soup == None:
            continue
        
        # Find the script tag with type="application/ld+json"
        script_tag = linked_page_soup.find('script', type='application/ld+json')

        if not script_tag:
            print("Script Tag Not Found")
            continue
        
        # Extract the content of the script tag
        script_content = script_tag.string.strip()

        # Parse the script content
        script_data = json.loads(script_content)

        if isinstance(script_data, list):
            script_data = script_data[0] if script_data else {}

        headline = script_data.get('headline')
        date_published_str = script_data.get('datePublished')
        if date_published_str == None:
            print('Date Published Not Found')
            continue
        
        date_published = datetime.strptime(date_published_str,'%Y-%m-%dT%H:%M:%S.%fz')

        author = script_data.get('author')
        author_name = ''
        if len(author) != 0:
            author_name = author[0].get('name') 
            
        thumbnail_url = script_data.get('thumbnailUrl')
        category= script_data.get('articleSection')
        if len(category) != 0:
            category = category[0]

        print('Headline: ', headline)
        print("Date Published: ", date_published)
        print("Author Name: ", author_name)
        print("Thumbnail URL: ", thumbnail_url)

        elements_with_tags = linked_page_soup.find_all('p', class_='paragraph')

        if not elements_with_tags:
            print("Article content Not Found")
            continue

        # Apply the remove_unwanted_attributes function to all tags 
        for element_with_tag in elements_with_tags:
            if element_with_tag is not None:
                remove_unwanted_attributes(element_with_tag)
                remove_empty_tags(element_with_tag)
                remove_whitespace(element_with_tag)

        news_content_tag = '\n '.join([str(paragraph) for paragraph in elements_with_tags if paragraph.get_text(strip=True).strip() != ''])

        print('Content with tag:', news_content_tag)
        
        # Remove all tags from the soup
        news_content = '\n'.join([paragraph.get_text(strip=True,separator=' ') for paragraph in elements_with_tags if paragraph.get_text(strip=True).strip() != ''])

        print("\nContent without tag:", news_content)
        print("\n\n\n")

        news.append(News(
            headline=headline,
            slug=slugify(headline),
            content=news_content,
            content_html=html.escape(news_content_tag),
            source='CNN',
            source_url=linked_page,
            image_url=thumbnail_url,
            category=category.capitalize(),
            author=author_name,
            published_date=date_published,
        ))

    return news

def store_news_in_chroma(news: List[News]):
    startTime = time.time()
    global chroma_client
    print("Connecting to ChromaDB...")

    ENVIRONMENT = os.getenv('ENVIRONMENT')

    CHROMA_HOST = os.getenv('CHROMA_HOST_LOCAL') if ENVIRONMENT == 'local' else os.getenv('CHROMA_HOST_PROD')

    CHROMA_PORT = os.getenv('CHROMA_PORT_LOCAL') if ENVIRONMENT == 'local' else os.getenv('CHROMA_PORT_PROD')

    ALLOW_RESET = os.getenv('ALLOW_RESET')

    SSL = False if ENVIRONMENT == 'local' else True
    
    chroma_client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=int(CHROMA_PORT), 
        settings=Settings(allow_reset=bool(ALLOW_RESET), anonymized_telemetry=False),
        ssl=SSL,
        headers={'authorization': 'PASS'}
    )
    
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    CHROMA_OPENAI_MODEL = os.getenv('CHROMA_OPENAI_MODEL')
    open_ai_embedding = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=CHROMA_OPENAI_MODEL
    )

    CHROMA_COLLECTION = os.getenv('CHROMA_COLLECTION')
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=open_ai_embedding)

    # split it into chunks
    text_splitter = CharacterTextSplitter(
        separator = " ",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
        is_separator_regex = False,
    )

    for _, article in enumerate(news):
        metadatas = [{ 'source': article.source, 'slug': article.slug, 'date': article.published_date.strftime('%Y-%m-%d %H:%M:%S'), 'headline' : article.headline}]
        documents = text_splitter.create_documents([article.content],metadatas=metadatas)

        exist = collection.query(
            query_texts=[documents[0].page_content],
            n_results=1,
            where={"slug": article.slug},
        )
        
        if len(exist['ids'][0]) > 0:
            print("Document already exist")
            continue
        
        for doc in documents:
            collection.add(
                ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
            )
    print('Records inserted successfully into ChromaDB')
    endTime = time.time()
    print(f"Time taken to insert records: {endTime - startTime} seconds")

def scrape_website():
    existing_slugs = get_existing_slugs_from_postgres()

    url_investing_website = 'https://www.investing.com/news/economy'
    newsInvesting = scrape_from_investing_website(url_investing_website, existing_slugs)

    url_cnbc_website = 'https://www.cnbc.com/economy'
    newsCnbc = scrape_from_cnbc_website(url_cnbc_website, existing_slugs)

    url_cnn_website = 'https://edition.cnn.com/business/economy'
    newsCnn = scrape_from_cnn_website(url_cnn_website, existing_slugs)

    all_news = newsInvesting + newsCnbc + newsCnn
    store_news_in_chroma(all_news) 
    store_news_in_postgres(all_news)

def main():
    load_dotenv()
    scrape_website()

if __name__=="__main__": 
    main()
    