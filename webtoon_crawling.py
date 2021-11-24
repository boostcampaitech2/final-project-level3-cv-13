
import argparse
from bs4 import BeautifulSoup
import glob
import os
import re
import requests

HEADERS = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'}
WT_LINK = "https://comic.naver.com"
N_PAGES = 1 # 각 웹툰 당 수집할 페이지 수

def arg_parse():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-l', '--webtoon_list', nargs='+', help='A list of webtoons to collect') 
    # nargs: '+' -> 하나의 args로 들어오는 것에 여러 개 요소를 받아줄 수 있음
    parser.add_argument("--webtoon_list", type=str, help="webtoon list to collect (in text file)")

    args = parser.parse_args()

    return args


def get_name_link(wt):
    """
    웹툰 하나에 대해 파싱 결과를 넘겨주면, 
    결과 내에서 다시 name과 link를 파싱해서 넘겨줌
    """
    link = WT_LINK+wt['href']
    name = wt.select_one('img')['title']

    return name, link


def get_all_name_link(wt_name_list):
    """
    웹툰 이름 리스트를 받으면,
    웹툰 메인에서 해당 웹툰의 이름과 link를 받아옴
    """
    wt_dict = {}
    # 웹툰 메인 불러오기
    html = requests.get(url="https://comic.naver.com/webtoon/weekday",headers=HEADERS)
    result = BeautifulSoup(html.content, "html.parser")
    wt_list = result.select("div.thumb > a")
    
    for wt in wt_list:
        name, link = get_name_link(wt)

        if name in wt_name_list:
            wt_dict[name] = link
    
    print("dictionary for webtoon urls built!")

    return wt_dict
        

def get_single_ep(ep_link, img_dir):
    """
    하나의 회차 크롤링

    - Args
        ep_link: a link for an episode
        img_dir: a directory to save collected images
    """
    # html 페이지 읽어오기
    html = requests.get(ep_link, headers = HEADERS)
    result = BeautifulSoup(html.content, "html.parser")

    # image 가져오기
    wt_images = result.select("div.wt_viewer > img")
    
    if len(os.listdir(img_dir)) == 0:
        num = 1
    else:
        imgs = glob.glob(img_dir+"/*.jpg")
        num = max(map(lambda x: int(re.search("[0-9]+", x).group()), imgs)) + 1 # 기존에 파일이 있는 경우 num 카운트를 이어나감


    for img in wt_images:
        img_name = os.path.join(img_dir, f"{num}.jpg")
        with open(img_name, "wb") as f:
            src = requests.get(img['src'], headers = HEADERS)
            f.write(src.content)
        
        num += 1

    

def crawl_webtoon(wt_name, wt_url):
    """
    하나의 웹툰에 대해 크롤링
    """
    img_dir = f"./images/{wt_name}"
    max_pages = N_PAGES

    # 웹툰명으로 폴더 생성
    os.makedirs(img_dir, exist_ok = True)
    print(f"A folder \"{wt_name}\" created.")

    # 지정된 페이지만큼 가져오기
    for pg_idx in range(N_PAGES):

        if pg_idx > max_pages-1: # 지정한 페이지 수가 실제 웹툰의 페이지 수보다 클 때의 예외처리
            break

        pg_url = wt_url+f"&page={pg_idx+1}"
        html = requests.get(pg_url, headers=HEADERS)
        result = BeautifulSoup(html.content, "html.parser")

        if pg_idx == 0:
            max_pages = int(result.select("em.num_page")[-1].text)
            
        episodes = list(map(lambda x: WT_LINK+x['href'], result.select("td.title > a")))
        
        for episode in episodes:
            get_single_ep(episode, img_dir)

    print(f"Collecting images of {wt_name} completed.")


def crawl_webtoon_all(wt_name_list):
    wt_dict = get_all_name_link(wt_name_list)

    for name, link in wt_dict.items():
        print(f"collecting images of {name} has been started.")
        crawl_webtoon(name, link)
    
    print("Done for all requests!")
    

def main():
    args = arg_parse()
    with open(args.webtoon_list, "r") as f:
        wt_name_list = f.readlines()
    wt_name_list = list(map(lambda x: x.strip(), wt_name_list))
    print("webtoons to collect: ", wt_name_list)
    crawl_webtoon_all(wt_name_list)


if __name__ == "__main__":
    main()

    