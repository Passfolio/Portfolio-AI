import os
import re
import json
from typing import List, Dict
from rapidfuzz import process
from docling.document_converter import DocumentConverter

# --- [전처리 함수: 노이즈 제거] ---
def super_clean_text(text: str) -> str:
    # --- 추가: 특정 중복 문구 제거 ---
    # 1. 불필요한 구분선 제거 (---, ===, |||)
    text = re.sub(r'[\-\|=]{3,}', '', text)
    
    # 2. 줄바꿈 정리 및 중복 단어 제거
    lines = text.split('\n')
    refined_lines = []
    for line in lines:
        words = line.split('|')
        new_words = []
        seen = set()
        for w in words:
            clean_w = w.strip()
            if clean_w and clean_w not in seen:
                new_words.append(clean_w)
                seen.add(clean_w)
        if new_words:
            refined_lines.append(" | ".join(new_words))
            
    return "\n".join(refined_lines)

# --- [핵심: 엔티티 기반 텍스트 분할기] ---

def split_by_entity_and_chunk(markdown_text: str, max_chars: int = 3000) -> List[Dict]:
    # 1. 표준 섹션 정의 및 유의어/오타 매핑 (띄어쓰기 없는 상태 기준)
    # 이력서 양식이 달라도 표준화된 엔티티(Key)로 묶어 LLM이 이해하기 쉽게 만듭니다.
    section_mapping = {
        "인적사항": ["인적사항", "인적사", "기본정보", "프로필", "personalinfo"],
        "학력사항": ["학력사항", "학력", "education", "출신학교", "학력및전공"],
        "병역사항": ["병역", "병역사항", "군필여부", "military", "병역관계"],
        "경력사항": ["경력", "경력사항", "업무경험", "experience", "실무경력"],
        "수상및활동": ["수상및경력", "수상", "대외활동", "프로젝트", "수상내역", "주요활동"],
        "자격및어학": ["외국어", "어학", "자격", "자격증", "면허", "어학능력", "license", "language"],
        "자기소개서": ["자소서", "자기소개서", "coverletter", "자기소개"]
    }

    # rapidfuzz 비교를 위해 모든 유의어를 하나의 리스트로 풀기 (평탄화)
    all_aliases = []
    alias_to_standard = {}
    for standard_key, aliases in section_mapping.items():
        for alias in aliases:
            all_aliases.append(alias)
            alias_to_standard[alias] = standard_key

    lines = markdown_text.split('\n')
    sections = {}
    current_section = "기타_상단" # 시작 기본값 (어떤 섹션인지 모를 앞부분)

    for line in lines:
        if not line.strip(): continue
        # 행의 앞부분에서 기호 및 공백 모두 제거 (비교를 위함)
        # 이력서에서 자주 쓰이는 특수문자(□, ■, ○, ●, · 등)도 함께 제거
        pure_line = re.sub(r'[\#\|\-\s□■○●·]+', '', line).strip()
        if not pure_line: continue
        # 앞의 10글자 정도만 잘라서 섹션 헤더인지 검사 (내용 전체를 비교하지 않기 위함)
        header_candidate = pure_line[:10].lower()

        # found_standard_key = None

        best_match_alias = None
        # 1차: 단순 시작 문자열 일치 검사 (정확한 매칭 우선)
        for alias in all_aliases:
            if header_candidate.startswith(alias):
                best_match_alias = alias
                break
        # 2차: 퍼지 매칭 (오타 교정 - 80% 이상 유사할 경우)
        if not best_match_alias and len(header_candidate) >= 2:
            # header_candidate와 가장 유사한 alias 찾기
            result = process.extractOne(header_candidate, all_aliases, score_cutoff=80)
            
            if result:
                best_match_alias = result[0] # 매칭된 문자열

        if best_match_alias:
            # 매칭된 유의어를 표준 엔티티 이름으로 변환
            # found_standard_key = alias_to_standard[best_match_alias]
            # current_section = found_standard_key

            current_section = alias_to_standard[best_match_alias]

            # 행에서 키워드 부분만 제거 (중복 제거)
            # 원본 line에서 매칭된 의미 단위(예: "학 력 사 항")를 정규식으로 안전하게 지움
            original_keyword_chars = list(best_match_alias)
            # 글자 사이에 공백이나 기호가 있을 수 있음을 감안한 패턴 생성
            pattern = r'[\s\|\-\#]*'.join(original_keyword_chars)
            # 대소문자 무시하고 맨 앞쪽 1번만 치환
            line = re.sub(pattern, '', line, count=1, flags=re.IGNORECASE).strip()
            # 찌꺼기 기호 제거
            line = re.sub(r'^[\#\|\-\s]+', '', line).strip()

        if current_section not in sections:
            sections[current_section] = []
        if line.strip(): # 키워드 떼고 남은 알맹이가 있으면 저장
            sections[current_section].append(line.strip())

    # 2. 같은 섹션끼리 뭉치고 청킹 처리
    final_chunks = []
    for section_name, content_list in sections.items():
        full_content = "\n".join(content_list).strip()
        if not full_content: continue

        if section_name == "자기소개서":
            final_chunks.append({
                "entity": section_name,
                "content": full_content,
                "is_chunked": False
            }) 
        else:
            if len(full_content) > max_chars:
                for i in range(0, len(full_content), max_chars):
                    final_chunks.append({
                        "entity": section_name,
                        "content": full_content[i:i+max_chars],
                        "is_chunked": True
                    })
            else:
                final_chunks.append({
                    "entity": section_name,
                    "content": full_content,
                    "is_chunked": False
                })   
    return final_chunks
