"""
exporters/portfolio_gen_pdf.py
──────────────────────────────────────────────────────────
run_cover_letter_to_portfolio() 결과 → PDF 저장

레이아웃 (섹션별):
  카테고리  (회색 소문자)
  프로젝트명  (볼드 대형)
  ── 메타 정보 (기간 / 역할 / 팀 / 기술스택) ──
  [ 개요 ]          본문
  [ 개발 ]          본문
  [ 이슈 ]          본문
  [ 성과 ]          본문
  ── 이미지 추천 ──  제안 내용
  ── 보완 요청 ──   gaps 목록
"""

from __future__ import annotations

from pathlib import Path
from fpdf import FPDF

_ASSETS_DIR   = Path(__file__).parent.parent.parent / "assets" / "fonts"
_FONT_REGULAR = str(_ASSETS_DIR / "NanumGothic.ttf")
_FONT_BOLD    = str(_ASSETS_DIR / "NanumGothicBold.ttf")
_FONT_FAMILY  = "Korean"

_SUB_LABELS = {
    "overview":    "개요",
    "development": "개발",
    "issue":       "이슈 및 해결",
    "result":      "성과",
}


class _PDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.add_font(_FONT_FAMILY, "",  _FONT_REGULAR)
        self.add_font(_FONT_FAMILY, "B", _FONT_BOLD)
        self.set_margins(20, 20, 20)
        self.set_auto_page_break(auto=True, margin=20)

    def _set(self, size: int, bold: bool = False, color: tuple = (30, 30, 30)):
        self.set_font(_FONT_FAMILY, "B" if bold else "", size)
        self.set_text_color(*color)

    def _gap(self, h: float = 4):
        self.ln(h)

    def _hr(self, color: tuple = (180, 180, 180), lw: float = 0.2):
        self.set_draw_color(*color)
        self.set_line_width(lw)
        self.line(self.get_x(), self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def _mc(self, text: str, size: int, bold: bool = False,
            color: tuple = (30, 30, 30), align: str = "L", lh: float | None = None):
        self._set(size, bold, color)
        line_h = lh if lh else size * 0.55
        self.multi_cell(0, line_h, text, align=align, new_x="LMARGIN", new_y="NEXT")

    def section_header(self, r: dict, idx: int, total: int):
        # 카테고리 + 번호
        self._mc(f"[{idx} / {total}]   {r.get('category', '')}", size=8, color=(140, 140, 140))
        self._gap(2)

        # 프로젝트명
        self._mc(r["project"] or r["section"], size=16, bold=True, color=(15, 15, 15))
        self._gap(1)
        self.set_draw_color(40, 40, 40)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), self.w - self.r_margin, self.get_y())
        self._gap(4)

        # 메타 정보
        meta_parts = []
        if r.get("period"):      meta_parts.append(f"기간: {r['period']}")
        if r.get("role"):        meta_parts.append(f"역할: {r['role']}")
        if r.get("team"):        meta_parts.append(f"팀:   {r['team']}")
        if r.get("tech_stack"):  meta_parts.append(f"기술: {', '.join(r['tech_stack'])}")

        if meta_parts:
            self._set(8, color=(90, 90, 90))
            for part in meta_parts:
                self.multi_cell(0, 4.5, f"  {part}", new_x="LMARGIN", new_y="NEXT")
            self._gap(5)
            self._hr(color=(200, 200, 200))
            self._gap(2)

    def sub_section_block(self, key: str, text: str):
        if not text:
            return
        label = _SUB_LABELS.get(key, key)
        self._mc(f"[ {label} ]", size=9, bold=True, color=(50, 80, 140))
        self._gap(1)
        self._mc(text, size=10, color=(25, 25, 25), align="L", lh=5.5)
        self._gap(5)
        self._hr(color=(210, 210, 210))
        self._gap(3)

    def image_suggestion_block(self, suggestion: str):
        if not suggestion:
            return
        self._gap(2)
        self._hr(color=(80, 140, 80), lw=0.3)
        self._gap(1)
        self._mc("[ 이미지 추천 ]", size=9, bold=True, color=(40, 110, 40))
        self._gap(1)
        self._mc(suggestion, size=9, color=(50, 90, 50), lh=5)
        self._gap(4)

    def eval_block(self, eval_result: dict | None):
        if not eval_result:
            return
        _LABELS  = {"A": "과정/판단력", "B": "역할/기여도", "C": "성과/인사이트", "D": "작성품질", "E": "직무연관성"}
        _WEIGHTS = {"A": 35, "B": 25, "C": 20, "D": 10, "E": 10}

        weighted = eval_result.get("weighted", 0)
        if   weighted >= 85: grade = "우수"
        elif weighted >= 70: grade = "양호"
        elif weighted >= 55: grade = "보통"
        else:                grade = "미흡"

        self._gap(2)
        self._hr(color=(60, 80, 140), lw=0.4)
        self._gap(1)
        self._mc(f"[ 포트폴리오 평가 ]  {weighted:.1f} / 100   {grade}", size=10, bold=True, color=(40, 55, 120))
        self._gap(2)

        llm = eval_result.get("llm", {})
        for key in ("A", "B", "C", "E"):
            item   = llm.get(key, {})
            score  = item.get("score", 0)
            reason = item.get("reason", "")
            fix    = item.get("fix", "")
            self._set(8, bold=True, color=(50, 65, 130))
            self.multi_cell(0, 4.5, f"  [{key}] {_LABELS[key]} ({_WEIGHTS[key]}%)  {score}/100",
                            new_x="LMARGIN", new_y="NEXT")
            self._set(8, color=(70, 80, 140))
            if reason:
                self.multi_cell(0, 4.5, f"      근거: {reason}", new_x="LMARGIN", new_y="NEXT")
            if fix:
                self.multi_cell(0, 4.5, f"      개선: {fix}", new_x="LMARGIN", new_y="NEXT")
            self._gap(1)

        d_info    = llm.get("D", {})
        d_score   = max(0, d_info.get("d1_volume", 0) + d_info.get("d2_quant", 0) + d_info.get("d3_penalty", 0))
        d_reason  = d_info.get("reason", "")
        d_fix     = d_info.get("fix", "")
        d1        = d_info.get("d1_volume", 0)
        d2        = d_info.get("d2_quant", 0)
        d3        = d_info.get("d3_penalty", 0)
        self._set(8, bold=True, color=(50, 65, 130))
        self.multi_cell(0, 4.5, f"  [D] {_LABELS['D']} ({_WEIGHTS['D']}%)  {d_score}/80",
                        new_x="LMARGIN", new_y="NEXT")
        self._set(8, color=(70, 80, 140))
        self.multi_cell(0, 4.5, f"      D1 분량: {d1}/40  |  D2 수치성과: {d2}/40  |  D3 감점: {d3}",
                        new_x="LMARGIN", new_y="NEXT")
        if d_reason:
            self.multi_cell(0, 4.5, f"      근거: {d_reason}", new_x="LMARGIN", new_y="NEXT")
        if d_fix:
            self.multi_cell(0, 4.5, f"      개선: {d_fix}", new_x="LMARGIN", new_y="NEXT")
        self._gap(3)

    def gaps_block(self, gaps: list[dict]):
        if not gaps:
            return
        self._gap(2)
        self._hr(color=(180, 100, 40), lw=0.3)
        self._gap(1)
        self._mc(f"[ 보완 요청사항 — {len(gaps)}건 ]", size=9, bold=True, color=(160, 80, 20))
        self._gap(2)
        for g in gaps:
            self._set(8, bold=True, color=(140, 70, 20))
            self.multi_cell(0, 4.5, f"  [{g['field']}]", new_x="LMARGIN", new_y="NEXT")
            self._set(8, color=(100, 60, 20))
            self.multi_cell(0, 4.5, f"    이유: {g['reason']}", new_x="LMARGIN", new_y="NEXT")
            self.multi_cell(0, 4.5, f"    요청: {g['user_action']}", new_x="LMARGIN", new_y="NEXT")
            if g.get("example"):
                self._set(8, bold=True, color=(80, 50, 140))
                self.multi_cell(0, 4.5, f"    참고 예시:", new_x="LMARGIN", new_y="NEXT")
                self._set(8, color=(80, 60, 160))
                self.multi_cell(0, 4.5, f"    {g['example']}", new_x="LMARGIN", new_y="NEXT")
            self._gap(2)
        self._gap(2)


def save_generated_portfolio_pdf(results: list[dict], output_path: str) -> str:
    """자소서 → 포트폴리오 생성 결과 → PDF 저장."""
    pdf   = _PDF()
    total = len(results)

    for i, r in enumerate(results, 1):
        pdf.add_page()
        pdf.section_header(r, i, total)

        for key in ("overview", "development", "issue", "result"):
            pdf.sub_section_block(key, r.get(key, ""))

        pdf.eval_block(r.get("eval"))
        pdf.image_suggestion_block(r.get("image_suggestion", ""))
        pdf.gaps_block(r.get("gaps", []))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(output_path)
    print(f"PDF 저장 완료 → {output_path}")
    return output_path
