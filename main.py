import os
import json
import time
from pathlib import Path
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schem import ResumeExtraction
from docling.document_converter import DocumentConverter

# 사용자의 로직 가져오기
from preprocess import super_clean_text, split_by_entity_and_chunk
from db_loader import ResumeDataLoader

# 환경 변수 로드
load_dotenv()

# 팀원(a.py)의 방식대로 genai Client 초기화
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# 팀원(a.py) 방식처럼 분당 요청 제한(RPM) 준수를 위한 딜레이 설정
RPM_DELAY = 13 

def extract_with_gemini(context_text: str, retries: int = 5) -> dict:
    """
    제공된 전체 텍스트를 분석하여 이력서(인적사항, 학력, 경력 등)와
    자기소개서 정보를 하나의 통합 스키마(ResumeExtraction)로 추출합니다.
    """
    prompt = f"""당신은 이력서 데이터 추출 전문가이자 자기소개서 분석 전문가입니다. 
이력서는 제공된 텍스트에서 지원자의 '사실 정보(Fact)'만 추출하여 JSON 구조로 만드세요. 
인적사항, 학력, 경력, 자격증, 수상내역, 병역 데이터에 집중하세요.

자기소개서에서는 제공된 텍스트에서 지원동기, 직무역량, 문제해결경험, 협업태도, 성격의 장단점, 취미, 희망경로(포부), 프로젝트 경험 등 모든 세부 문항을 빠짐없이 찾아내어 분석하세요.
데이터가 명시적으로 구분되어 있지 않더라도 내용의 의미에 따라 적절한 필드에 배치하세요.

[데이터 원본]
{context_text}"""

    for attempt in range(retries):
        try:
            # 팀원의 a.py와 동일하게 response_schema를 사용하여 JSON 출력 강제
            response = client.models.generate_content(
                model="gemini-3-flash-preview", 
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ResumeExtraction,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait_time = RPM_DELAY * (attempt + 1)
                print(f"    ⚠ Rate limit 발생. {wait_time}초 대기 후 재시도...")
                time.sleep(wait_time); continue
            else:
                print(f"❌ Gemini 호출 중 오류 발생: {e}")
                raise

# ── 메인 실행부 ──────────────────────────────────────────────────────
def main():
    # DB 설정
    db_config = {
        "host": "127.0.0.1",
        "database": "postgres",
        "user": "sangbin",
        "password": "", # 필요시 설정
        "port": 5432
    }

    # 처리할 PDF 리스트 (여러 파일 처리 가능하도록 구조화)
    pdf_paths = [
        "/Users/sangbin/Downloads/software_gabia.pdf" 
    ]
    
    if not api_key:
        print("❌ API 키가 설정되지 않았습니다. .env 파일에 GEMINI_API_KEY를 설정해주세요.")
        return

    converter = DocumentConverter()

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"❌ 파일을 찾을 수 없습니다: {pdf_path}")
            continue

        print(f"\n{'='*60}")
        print(f"파일 처리 시작: {Path(pdf_path).name}")
        print('='*60)

        # 1. 추출 (Docling)
        print("🚀 Step 1: Docling으로 PDF 추출 중...")
        try:
            doc_result = converter.convert(pdf_path)
            markdown_output = doc_result.document.export_to_markdown()
        except Exception as e:
            print(f"❌ PDF 추출 실패: {e}")
            continue
        
        # 2. 전처리 (사용자의 preprocess.py 로직)
        print("🔪 Step 2: 전처리 및 섹션 분할 중...")
        cleaned_md = super_clean_text(markdown_output)
        processed_chunks = split_by_entity_and_chunk(cleaned_md)

        # 3. 텍스트 그룹화 (섹션 태그 포함)
        full_context = ""
        for chunk in processed_chunks:
            chunk_text = f"### {chunk['entity']} ###\n{chunk['content']}\n\n"
            full_context += chunk_text

        # 4. LLM 변환 (팀원의 a.py 방식)
        print("🤖 Step 3: Gemini(JSON Mode) 호출 중...")
        try:
            result_json = extract_with_gemini(full_context)
            print("✅ 추출 완료")
            
            # 5. DB 적재 (임베딩 포함)
            print("💾 Step 4: 임베딩(BGE-M3) 및 PostgreSQL 적재 시작...")
            loader = ResumeDataLoader(db_config)
            try:
                loader.connect_db()
                chunks = loader.prepare_data(result_json)
                loader.insert_data(chunks)
            finally:
                loader.close()
            
        except Exception as e:
            print(f"❌ 처리 중 실패: {e}")

        # 팀원 방식처럼 분당 요청 제한(RPM) 준수를 위해 대기
        print(f"\n⏳ RPM 제한 준수를 위해 {RPM_DELAY}초 대기 중... (Rate Limit 방지)")
        time.sleep(RPM_DELAY)

if __name__ == "__main__":
    main()
