import json
from collections import Counter

done = json.load(open('output/mock_cover_letters_ctx.json', encoding='utf-8'))

# 기본 통계
total = len(done)
has_context = sum(1 for c in done if c.get('context'))
sources = {c['source'] for c in done}
categories = Counter(c.get('category', 'unknown') for c in done)
sections = Counter(c.get('section', 'unknown') for c in done)

# 텍스트 길이
ctx_lens = [len(c.get('context', '')) for c in done if c.get('context')]
txt_lens = [len(c.get('text', '')) for c in done]
twc_lens = [len(c.get('text_with_context', '')) for c in done if c.get('text_with_context')]

print(f'=== 기본 통계 ===')
print(f'총 청크 수: {total}')
print(f'context 존재: {has_context} ({has_context/total*100:.1f}%)')
print(f'고유 source 수: {len(sources)}')
print()
print(f'=== 카테고리 분포 ===')
for cat, cnt in categories.most_common():
    print(f'  {cat}: {cnt}')
print()
print(f'=== 섹션 분포 (상위 10) ===')
for sec, cnt in sections.most_common(10):
    print(f'  {sec}: {cnt}')
print()
print(f'=== 텍스트 길이 (문자 수) ===')
print(f'  context    avg={sum(ctx_lens)/len(ctx_lens):.0f}  min={min(ctx_lens)}  max={max(ctx_lens)}')
print(f'  text       avg={sum(txt_lens)/len(txt_lens):.0f}  min={min(txt_lens)}  max={max(txt_lens)}')
if twc_lens:
    print(f'  with_ctx   avg={sum(twc_lens)/len(twc_lens):.0f}  min={min(twc_lens)}  max={max(twc_lens)}')
print()

# 샘플 출력
sample = next((c for c in done if c.get('context')), None)
if sample:
    print(f'=== 샘플 청크 ===')
    print(f'id: {sample["id"]}')
    print(f'category: {sample.get("category")}')
    print(f'section: {sample.get("section")}')
    print(f'context: {sample["context"][:200]}')
    print(f'text: {sample["text"][:200]}')
