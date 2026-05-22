"""
exporters/cover_letter_pdf.py
──────────────────────────────────────────────────────────
run_portfolio_to_cover_letter() 결과 → PDF 저장

레이아웃 (섹션별):
  [문항]  회색 소문자
  (빈줄)
  소제목  볼드 대형
  (빈줄)
  본문
  ── 평가 ──────────────────────────────────
  총점 / 등급
  A·B·C·D·E 항목별 점수 + 설명
"""

from __future__ import annotations

from pathlib import Path
from fpdf import FPDF

# ── 폰트 설정 ──────────────────────────────────────────────────────────────────
_ASSETS_DIR     = Path(__file__).parent.parent.parent / "assets" / "fonts"
_FONT_REGULAR   = str(_ASSETS_DIR / "NanumGothic.ttf")
_FONT_BOLD      = str(_ASSETS_DIR / "NanumGothicBold.ttf")
_FONT_FAMILY    = "Korean"

# ── 평가 항목 메타 ─────────────────────────────────────────────────────────────
_LABELS = {
    "A": "지원동기",
    "B": "직무역량",
    "C": "인재상",
    "D": "작성품질",
    "E": "AI 의심도",
}
_WEIGHTS = {"A": 15, "B": 35, "C": 20, "D": 15, "E": 15}
_CRITERIA = {
    "A": "두괄식 구성 여부, 기업 이해도 반영, 입사 설득력",
    "B": "STAR 구조(상황→과제→행동→결과), 수치·정성 성과, 경험→성장→기여 연결",
    "C": "핵심 가치관 부합, 소통·협업 경험, 갈등 해결 사례",
    "D": "분량 충족률, 반복 문장, 수치 포함 여부 (규칙 기반 채점)",
    "E": "AI 특유 문체 패턴 감지 — 높을수록 감점 (역산 적용)",
}


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

    # 편의 래퍼
    def _set(self, size: int, bold: bool = False, color: tuple = (30, 30, 30)):
        style = "B" if bold else ""
        self.set_font(_FONT_FAMILY, style, size)
        self.set_text_color(*color)

    def _line(self, text: str, size: int, bold: bool = False,
              color: tuple = (30, 30, 30), ln: bool = True, align: str = "L"):
        self._set(size, bold, color)
        self.multi_cell(0, size * 0.55, text, align=align, new_x="LMARGIN", new_y="NEXT" if ln else "NEXT")

    def _gap(self, h: float = 4):
        self.ln(h)

    def _hr(self, color: tuple = (180, 180, 180)):
        self.set_draw_color(*color)
        self.line(self.get_x(), self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    # 섹션 렌더
    def add_section(self, result: dict, idx: int, total: int):
        label    = result["label"]
        question = result["question"]
        text     = result["text"]
        ev       = result.get("eval", {})
        weighted = ev.get("weighted", 0)
        llm      = ev.get("llm", {})
        d_info   = ev.get("D", {})

        # ── 문항 번호 + 문항 텍스트 ──────────────────────────────
        self._set(8, color=(140, 140, 140))
        self.multi_cell(0, 4.5, f"[문항 {idx}/{total}]  {question}",
                        align="L", new_x="LMARGIN", new_y="NEXT")
        self._gap(4)

        # ── 소제목 ───────────────────────────────────────────────
        self._line(label, size=15, bold=True, color=(20, 20, 20))
        self._gap(2)
        self._hr(color=(80, 80, 80))
        self._gap(1)

        # ── 본문 ─────────────────────────────────────────────────
        self._set(10, color=(30, 30, 30))
        self.multi_cell(0, 5.5, text, align="J", new_x="LMARGIN", new_y="NEXT")
        self._gap(3)
        self._line(f"{len(text)}자", size=8, color=(130, 130, 130))
        self._gap(6)

        # ── 평가 헤더 ─────────────────────────────────────────────
        self._hr(color=(100, 100, 100))
        self._set(9, bold=True, color=(60, 60, 60))
        grade = _grade(weighted)
        self.multi_cell(0, 5, f"평가 결과    총점  {weighted:.1f} / 100   {grade}",
                        align="L", new_x="LMARGIN", new_y="NEXT")
        self._gap(4)

        # ── A·B·C·E ───────────────────────────────────────────────
        for key in ["A", "B", "C", "E"]:
            item = llm.get(key, {})
            if not item:
                continue
            score = item.get("score", "-")

            if key == "E":
                e_adj = 100 - int(score) if isinstance(score, int) else "-"
                score_str = f"역산 {e_adj} / 100  (원점수 {score})"
            else:
                score_str = f"{score} / 10"

            # 항목 제목
            self._set(9, bold=True, color=(40, 40, 40))
            self.multi_cell(
                0, 5,
                f"{key}.  {_LABELS[key]}  ({_WEIGHTS[key]}%)   →   {score_str}",
                new_x="LMARGIN", new_y="NEXT",
            )
            # 평가 기준 설명
            self._set(8, color=(110, 110, 110))
            self.multi_cell(0, 4.5, f"   기준: {_CRITERIA[key]}",
                            new_x="LMARGIN", new_y="NEXT")

            # B 전용: STAR / 수치
            if key == "B":
                star = item.get("star_score", "-")
                num  = item.get("num_score", "-")
                self._set(8, color=(90, 90, 90))
                self.multi_cell(0, 4.5, f"   STAR 구조 {star}/10   |   수치 성과 {num}/10",
                                new_x="LMARGIN", new_y="NEXT")

            # 근거
            reason = item.get("reason", "")
            if reason:
                self._set(8, color=(60, 60, 60))
                self.multi_cell(0, 4.5, f"   근거: {reason}",
                                new_x="LMARGIN", new_y="NEXT")

            # 개선 (E 제외)
            if key != "E":
                fix = item.get("fix", "")
                if fix:
                    self._set(8, color=(80, 100, 140))
                    self.multi_cell(0, 4.5, f"   개선: {fix}",
                                    new_x="LMARGIN", new_y="NEXT")
            else:
                detected = item.get("detected", [])
                pat = ", ".join(detected) if detected else "없음"
                self._set(8, color=(60, 60, 60))
                self.multi_cell(0, 4.5, f"   감지 패턴: {pat}",
                                new_x="LMARGIN", new_y="NEXT")

            self._gap(3)

        # ── D (규칙 기반) ─────────────────────────────────────────
        d_score  = d_info.get("score", "-")
        d_detail = d_info.get("detail", {})

        self._set(9, bold=True, color=(40, 40, 40))
        self.multi_cell(0, 5, f"D.  {_LABELS['D']}  ({_WEIGHTS['D']}%)   →   {d_score} / 10  [규칙 기반]",
                        new_x="LMARGIN", new_y="NEXT")
        self._set(8, color=(110, 110, 110))
        self.multi_cell(0, 4.5, f"   기준: {_CRITERIA['D']}",
                        new_x="LMARGIN", new_y="NEXT")

        if d_detail:
            length   = d_detail.get("length", "-")
            dup      = d_detail.get("dup_count", "-")
            has_num  = "✓" if d_detail.get("has_number") else "✗"
            vol      = d_detail.get("vol_status", "")

            if "char_limit" in d_detail:
                limit_str = f"제한 {d_detail['char_limit']}자  ({d_detail.get('fill_ratio', '-')}% 충족)"
            elif "free_standard" in d_detail:
                cat = "직무역량" if d_detail.get("is_competency") else "일반"
                limit_str = f"자유형식 ({cat} {d_detail['free_standard']}자 기준)"
            else:
                limit_str = ""

            detail_line = f"   {length}자   |   {limit_str}   |   상태: {vol}   |   반복: {dup}건   |   수치: {has_num}"
            self._set(8, color=(60, 60, 60))
            self.multi_cell(0, 4.5, detail_line, new_x="LMARGIN", new_y="NEXT")

        self._gap(3)

        # ── 총평 ─────────────────────────────────────────────────
        overall = llm.get("overall", "")
        if overall:
            self._set(8, bold=True, color=(60, 60, 60))
            self.multi_cell(0, 4.5, "총평", new_x="LMARGIN", new_y="NEXT")
            self._set(8, color=(60, 60, 60))
            self.multi_cell(0, 4.5, overall, new_x="LMARGIN", new_y="NEXT")

        # ── 이미지 제안 ───────────────────────────────────────────
        img_suggestion = result.get("image_suggestion", "")
        if img_suggestion:
            self._gap(4)
            self._hr(color=(180, 140, 60))
            self._set(8, bold=True, color=(160, 110, 30))
            self.multi_cell(0, 4.5, "포트폴리오 강화 이미지 제안", new_x="LMARGIN", new_y="NEXT")
            self._gap(1)
            self._set(8, color=(140, 100, 30))
            self.multi_cell(0, 4.5, img_suggestion, new_x="LMARGIN", new_y="NEXT")


def save_cl_improvement_pdf(results: list[dict], output_path: str) -> str:
    """자소서 수정 결과 (run_from_pdf) → PDF 저장.

    레이아웃: 섹션 제목 → 원문(회색) → 개선안(검정) → 변경사항 → 전후 점수
    """
    pdf   = _PDF()
    total = len(results)

    for i, r in enumerate(results, 1):
        pdf.add_page()

        # ── 섹션 제목 ────────────────────────────────────────────
        pdf._set(8, color=(130, 130, 130))
        pdf.multi_cell(0, 4.5, f"[{i} / {total}]   {r.get('category', '')}",
                       new_x="LMARGIN", new_y="NEXT")
        pdf._gap(2)
        pdf._set(15, bold=True, color=(20, 20, 20))
        pdf.multi_cell(0, 8, r["section"], new_x="LMARGIN", new_y="NEXT")
        pdf._gap(1)
        pdf._hr(color=(80, 80, 80))
        pdf._gap(2)

        # ── 원문 ────────────────────────────────────────────────
        pdf._set(8, bold=True, color=(120, 120, 120))
        pdf.multi_cell(0, 5, f"원문  ({len(r['original'])}자)", new_x="LMARGIN", new_y="NEXT")
        pdf._gap(1)
        pdf._set(9, color=(140, 140, 140))
        pdf.multi_cell(0, 5, r["original"], align="J", new_x="LMARGIN", new_y="NEXT")
        pdf._gap(5)

        # ── 개선안 ───────────────────────────────────────────────
        pdf._set(8, bold=True, color=(30, 100, 180))
        pdf.multi_cell(0, 5, f"개선안  ({len(r['improved'])}자)", new_x="LMARGIN", new_y="NEXT")
        pdf._hr(color=(30, 100, 180))
        pdf._gap(1)
        pdf._set(10, color=(20, 20, 20))
        pdf.multi_cell(0, 5.5, r["improved"], align="J", new_x="LMARGIN", new_y="NEXT")
        pdf._gap(5)

        # ── 주요 변경사항 ─────────────────────────────────────────
        changes = r.get("changes", [])
        if changes:
            pdf._set(8, bold=True, color=(60, 60, 60))
            pdf.multi_cell(0, 5, "주요 변경사항", new_x="LMARGIN", new_y="NEXT")
            pdf._set(8, color=(70, 70, 70))
            for ch in changes:
                pdf.multi_cell(0, 4.5, f"  •  {ch}", new_x="LMARGIN", new_y="NEXT")
            pdf._gap(4)

        # ── 전후 점수 ─────────────────────────────────────────────
        before = r.get("eval_before")
        after  = r.get("eval_after")
        delta  = r.get("eval_delta")
        if before is not None and after is not None:
            pdf._hr(color=(160, 160, 160))

            # 총점 헤더
            sign = "+" if delta >= 0 else ""
            pdf._set(10, bold=True, color=(30, 30, 30))
            pdf.multi_cell(
                0, 5.5,
                f"총점   {before:.1f}  →  {after:.1f}   ({sign}{delta:.1f})   "
                f"{_grade(before)}  →  {_grade(after)}",
                new_x="LMARGIN", new_y="NEXT",
            )
            pdf._gap(3)

            # 항목별 전후 비교표
            detail = r.get("eval_detail", {})
            if detail:
                col_w = (pdf.w - pdf.l_margin - pdf.r_margin) / 4

                # 헤더 행
                pdf._set(8, bold=True, color=(80, 80, 80))
                for header in ["항목", "원문", "개선안", "변화"]:
                    pdf.multi_cell(col_w, 5.5, header, align="C",
                                   new_x="RIGHT", new_y="TOP", border="B")
                pdf.ln(5.5)

                for key in ["A", "B", "C", "D", "E"]:
                    item = detail.get(key)
                    if not item:
                        continue
                    b = item.get("before", "-")
                    a = item.get("after", "-")
                    d = item.get("delta", 0)
                    sign2 = "+" if isinstance(d, (int, float)) and d > 0 else ""

                    if isinstance(d, (int, float)) and d > 0:
                        d_color = (0, 130, 70)
                    elif isinstance(d, (int, float)) and d < 0:
                        d_color = (180, 40, 40)
                    else:
                        d_color = (100, 100, 100)

                    label = f"{key}. {_LABELS[key]} ({_WEIGHTS[key]}%)"
                    pdf._set(8, color=(50, 50, 50))
                    pdf.multi_cell(col_w, 5.5, label,
                                   new_x="RIGHT", new_y="TOP")
                    pdf.multi_cell(col_w, 5.5, str(b), align="C",
                                   new_x="RIGHT", new_y="TOP")
                    pdf.multi_cell(col_w, 5.5, str(a), align="C",
                                   new_x="RIGHT", new_y="TOP")
                    pdf._set(8, color=d_color)
                    pdf.multi_cell(col_w, 5.5, f"{sign2}{d}", align="C",
                                   new_x="LMARGIN", new_y="NEXT")

                pdf._gap(4)

                # B 전용 세부: STAR / 수치 성과
                b_item = detail.get("B", {})
                if b_item:
                    rows = []
                    for timing, key2 in [("원문", "before_detail"), ("개선안", "after_detail")]:
                        d2 = b_item.get(key2, {})
                        if d2:
                            rows.append(f"{timing}  —  STAR {d2.get('star_score', '-')}/100  |  수치성과 {d2.get('num_score', '-')}/100")
                    if rows:
                        pdf._set(8, color=(90, 90, 90))
                        for row in rows:
                            pdf.multi_cell(0, 4.5, f"  B 세부: {row}", new_x="LMARGIN", new_y="NEXT")
                        pdf._gap(2)

                # D 세부: 글자수·충족률
                d_item = detail.get("D", {})
                for timing, key2 in [("원문", "before_detail"), ("개선안", "after_detail")]:
                    d2 = d_item.get(key2, {})
                    if d2:
                        length  = d2.get("length", "-")
                        vol     = d2.get("vol_status", "")
                        dup     = d2.get("dup_count", "-")
                        has_num = "O" if d2.get("has_number") else "X"
                        if "char_limit" in d2:
                            limit_str = f"제한 {d2['char_limit']}자 ({d2.get('fill_ratio', '-')}% 충족)"
                        elif "free_standard" in d2:
                            cat = "직무역량" if d2.get("is_competency") else "일반"
                            limit_str = f"{cat} {d2['free_standard']}자 기준"
                        else:
                            limit_str = ""
                        pdf._set(8, color=(90, 90, 90))
                        pdf.multi_cell(
                            0, 4.5,
                            f"  D {timing}: {length}자  |  {limit_str}  |  {vol}  |  반복 {dup}건  |  수치 {has_num}",
                            new_x="LMARGIN", new_y="NEXT",
                        )
                pdf._gap(3)

            # 개선 근거
            reasoning = r.get("reasoning", "")
            if reasoning:
                pdf._set(8, bold=True, color=(70, 70, 70))
                pdf.multi_cell(0, 4.5, "개선 근거", new_x="LMARGIN", new_y="NEXT")
                pdf._set(8, color=(90, 90, 90))
                pdf.multi_cell(0, 4.5, reasoning, new_x="LMARGIN", new_y="NEXT")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(output_path)
    print(f"PDF 저장 완료 → {output_path}")
    return output_path


def save_pdf(results: list[dict], output_path: str) -> str:
    """자소서 생성 결과 리스트 → PDF 저장.

    Args:
        results:     run_portfolio_to_cover_letter() 반환값
        output_path: 저장 경로 (.pdf)
    Returns:
        저장된 파일 경로
    """
    pdf   = _PDF()
    total = len(results)

    for i, result in enumerate(results, 1):
        pdf.add_page()
        pdf.add_section(result, i, total)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(output_path)
    print(f"PDF 저장 완료 → {output_path}")
    return output_path
