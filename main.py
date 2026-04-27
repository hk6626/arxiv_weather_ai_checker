import os
import time
import yaml
import feedparser
from github import Github, Auth
from google import genai

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_arxiv_entries(category):
    # ArXivのRSSフィードのURL
    rss_url = f"http://export.arxiv.org/rss/{category}"
    print(f"Fetching RSS feed from: {rss_url}")
    feed = feedparser.parse(rss_url)
    return feed.entries

def extract_arxiv_id(entry_link):
    # 例: http://arxiv.org/abs/2410.12345 -> 2410.12345
    return entry_link.split('/')[-1]

def issue_exists(g, repo_name, arxiv_id):
    """
    指定したArXiv IDを持つIssueがすでにリポジトリに存在するか確認する
    """
    query = f"repo:{repo_name} {arxiv_id} in:body"
    print(f"Searching GitHub issues with query: {query}")
    try:
        issues = g.search_issues(query=query)
        return issues.totalCount > 0
    except Exception as e:
        print(f"Warning: Failed to search issues. Assumes not exists. Error: {e}")
        # 検索に失敗した場合（API制限など）は安全のために重複とみなすか、
        # あるいはそのまま進めるか。ここでは暫定でFalseを返す。
        return False

def evaluate_and_summarize(client, entry, themes, model_name):
    """
    Gemini APIを使用して論文がテーマに合致するか判定し、要約を作成する
    """
    title = entry.title
    summary = entry.summary
    
    themes_str = "\n".join([f"- {t}" for t in themes])
    
    prompt = f"""
あなたは気象学、地球物理学、および機械学習の専門家です。
以下の論文のタイトルとアブストラクトを読み、下記の「関心のあるテーマ」に合致するか（または関連性が高いか）を判定してください。

【関心のあるテーマ】
{themes_str}

【論文情報】
タイトル: {title}
アブストラクト: {summary}

【指示】
もしこの論文が「関心のあるテーマ」のいずれにも関連しないと判断した場合は、以下の文字列のみを出力してください（他の文章は一切含めないでください）。
FALSE

もし関連性が高いと判断した場合は、論文の概要を日本語でわかりやすく要約してください。
出力は以下のMarkdown形式としてください。

### 関連するテーマ
（どのテーマに関連しているか）

### 論文の概要・ポイント
- （ポイント1）
- （ポイント2）
- （ポイント3）
...
"""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            text = response.text.strip()
            
            if text == "FALSE":
                return None
            return text
        except Exception as e:
            error_msg = str(e).lower()
            # 429 Too Many Requests や Resource Exhausted などのレートリミットエラー、503の一時的エラーを検知
            if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg or "too many requests" in error_msg or "503" in error_msg or "timeout" in error_msg or "overloaded" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = 30
                    print(f"Rate limit or temporary error hit. Waiting for {wait_time} seconds before retrying (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
            print(f"Error calling Gemini API for {title}: {e}")
            return None

def create_issue(repo, entry, arxiv_id, summary_text):
    title = f"[ArXiv] {entry.title}"
    # タイトルが長すぎる場合は切り詰める（GitHub Issueのタイトル上限に配慮）
    if len(title) > 200:
        title = title[:197] + "..."
        
    authors = getattr(entry, 'author', 'Unknown')
    link = getattr(entry, 'link', '')
    
    body = f"""## 論文情報
- **Title**: {entry.title}
- **Authors**: {authors}
- **URL**: {link}
- **ArXiv ID**: `{arxiv_id}`

## AIによる判定・要約 (Gemini)
{summary_text}

---
*※ このIssueはArXiv新着チェッカーにより自動生成されています。*
"""
    print(f"Creating Issue: {title}")
    repo.create_issue(title=title, body=body)

def main():
    config = load_config()
    
    category = config.get("arxiv_category", "physics.ao-ph")
    themes = config.get("search_themes", [])
    repo_name = config.get("github_repo")
    model_name = config.get("gemini_model", "gemini-1.5-flash")
    
    github_token = os.environ.get("GITHUB_TOKEN")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    if not github_token or not gemini_api_key:
        print("Error: GITHUB_TOKEN or GEMINI_API_KEY environment variables are not set.")
        return

    # Initialize APIs
    # timeoutを設定して10分間ハングするのを防ぐ
    client = genai.Client(api_key=gemini_api_key, http_options={'timeout': 60.0})
    auth = Auth.Token(github_token)
    g = Github(auth=auth)
    
    try:
        repo = g.get_repo(repo_name)
    except Exception as e:
        print(f"Error: Could not access repository '{repo_name}'. Please check the config and token permissions. {e}")
        return

    entries = get_arxiv_entries(category)
    print(f"Found {len(entries)} entries in the RSS feed.")
    
    # 最新のRSSフィードには最大で数十件含まれる。
    # APIのRate Limitを考慮して、少しスリープを入れながら処理する。
    processed_count = 0
    posted_count = 0
    
    for entry in entries:
        arxiv_id = extract_arxiv_id(entry.link)
        
        # 重複チェック
        if issue_exists(g, repo_name, arxiv_id):
            print(f"Skip: Issue already exists for {arxiv_id}")
            continue
            
        print(f"\nProcessing: {arxiv_id} - {entry.title}")
        
        # 判定と要約
        summary_text = evaluate_and_summarize(client, entry, themes, model_name)
        
        if summary_text:
            print(" -> Match found! Creating issue...")
            create_issue(repo, entry, arxiv_id, summary_text)
            posted_count += 1
            # Issue作成後、少し待機してAPI制限を回避
            time.sleep(2)
        else:
            print(" -> Not related. Skipping.")
            
        processed_count += 1
        
        # Gemini APIのレート制限に配慮 (無料枠は15RPMなど)
        # 短期間に大量のリクエストを送らないようにする
        time.sleep(4)

    print(f"\nDone! Processed {processed_count} new papers. Posted {posted_count} issues.")

if __name__ == "__main__":
    main()
