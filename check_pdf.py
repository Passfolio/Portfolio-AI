import fitz

doc = fitz.open(r"portfoliosample/박중헌_포트폴리오.pdf")
print(f"총 페이지 수: {len(doc)}")
for i in range(min(3, len(doc))):
    text = doc[i].get_text()
    print(f"\n=== 페이지 {i+1} === (텍스트 길이: {len(text)}자)")
    print(repr(text[:500]))
doc.close()
