"""
exporters/portfolio_pdf.py
──────────────────────────────────────────────────────────
run_portfolio_from_pdf() 결과 → PDF 저장

원본 레이아웃 유지:
  프로젝트명  (볼드 대형 헤더)
    └─ 서브섹션  (볼드 소헤더)
         원문 (회색)
         개선안 (검정)
         전후 점수
"""

from __future__ import annotations

from pathlib import Path
from fpdf import FPDF

_ASSETS_DIR   = Path(__file__).parent.parent / "assets" / "fonts"
_FONT_REGULAR = str(_ASSETS_DIR / "NanumGothic.ttf")
_FONT_BOLD    = str(_ASSETS_DIR / "NanumGothicBold.ttf")
_FONT_FAMILY  = "Korean"

_PF_LABELS = {
    "A": "과정/판단력",
    "B": "역할/기여도",
    "C": "성과/인사이트",
    "D": "작성품질",
    "E": "직무연관성",
}
_PF_WEIGHTS = {"A": 35, "B": 25, "C": 20, "D": 10, "E": 10}


def _grade(score: float) -> str:
    if   score >= 85: return "우수 ★★★"
    elif score >= 70: return "양호 ★★"
    elif score >= 55: return "보통 ★"
    else:             return "미흡"


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

    # ── 프로젝트 헤더 ────────────────────────────────────────────────────────
    def project_header(self, project: str, section: str):
        self._mc(section, size=8, color=(140, 140, 140))
        self._gap(1)
        self._mc(project or section, size=16, bold=True, color=(15, 15, 15))
        self._gap(1)
        self.set_draw_color(40, 40, 40)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    # ── 서브섹션 블록 ────────────────────────────────────────────────────────
    def sub_section_block(self, r: dict):
        if r.get("sub_section") == "이미지":
            return  # 이미지 청크는 수정 PDF에서 제외 (텍스트 개선 대상 아님)

        sub = r.get("sub_section", "")

        # 서브섹션 소헤더
        if sub:
            self._mc(f"[ {sub} ]", size=9, bold=True, color=(80, 80, 80))
            self._gap(1)

        # 원문
        self._mc(f"원문  ({len(r['original'])}자)", size=8,
                 bold=True, color=(130, 130, 130))
        self._gap(1)
        self._mc(r["original"], size=9, color=(155, 155, 155), align="J", lh=5)
        self._gap(4)

        # 개선안
        self._set(8, bold=True, color=(30, 100, 180))
        self.multi_cell(0, 5, f"개선안  ({len(r['improved'])}자)",
                        new_x="LMARGIN", new_y="NEXT")
        self._hr(color=(30, 100, 180), lw=0.3)
        self._gap(1)
        self._mc(r["improved"], size=10, color=(20, 20, 20), align="J", lh=5.5)
        self._gap(4)

        # 주요 변경사항
        changes = r.get("changes", [])
        if changes:
            self._mc("주요 변경사항", size=8, bold=True, color=(60, 60, 60))
            self._set(8, color=(70, 70, 70))
            for ch in changes:
                self.multi_cell(0, 4.5, f"  •  {ch}",
                                new_x="LMARGIN", new_y="NEXT")
            self._gap(3)

        # 전후 점수
        before = r.get("eval_before")
        after  = r.get("eval_after")
        delta  = r.get("eval_delta")
        if before is not None and after is not None:
            self._hr(color=(160, 160, 160))
            sign = "+" if delta >= 0 else ""
            self._mc(
                f"전후 점수   {before:.1f}  →  {after:.1f}  ({sign}{delta:.1f})   "
                f"{_grade(before)}  →  {_grade(after)}",
                size=9, bold=True, color=(40, 40, 40),
            )

            # 항목별 전후 상세 (eval_detail 있을 때)
            detail = r.get("eval_detail", {})
            if detail:
                self._gap(2)
                parts = []
                for key in ["A", "B", "C", "D", "E"]:
                    item = detail.get(key, {})
                    if item:
                        b = item.get("before", "-")
                        a = item.get("after", "-")
                        d = item.get("delta", 0)
                        sign2 = "+" if isinstance(d, (int, float)) and d >= 0 else ""
                        parts.append(f"{key}.{_PF_LABELS.get(key, key)}  {b}→{a}({sign2}{d})")
                if parts:
                    self._set(8, color=(90, 90, 90))
                    self.multi_cell(0, 4.5, "   " + "   |   ".join(parts),
                                    new_x="LMARGIN", new_y="NEXT")

        self._gap(6)
        self._hr(color=(210, 210, 210))
        self._gap(4)


def save_improvement_pdf(results: list[dict], output_path: str) -> str:
    """포트폴리오 수정 결과 (run_portfolio_from_pdf) → PDF 저장.

    원본 레이아웃 유지: 프로젝트 단위로 묶어 섹션→서브섹션 순으로 출력.
    """
    pdf = _PDF()

    # 프로젝트·섹션 순서 유지하며 그룹화
    ordered_groups: list[tuple[str, str, list[dict]]] = []
    seen: dict[tuple, int] = {}
    for r in results:
        key = (r.get("section", ""), r.get("project", ""))
        if key not in seen:
            seen[key] = len(ordered_groups)
            ordered_groups.append((key[0], key[1], []))
        ordered_groups[seen[key]][2].append(r)

    for section, project, items in ordered_groups:
        pdf.add_page()
        pdf.project_header(project, section)
        for r in items:
            pdf.sub_section_block(r)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(output_path)
    print(f"PDF 저장 완료 → {output_path}")
    return output_path
