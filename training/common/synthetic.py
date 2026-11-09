"""Synthetic dataset generators (clearly labeled, for pipeline testing only).

Every example has ``source="synthetic"``. These 20 examples per track
exist only to prove the pipeline runs end to end (schema validation,
training script startup, metric computation) with NO real data yet.
Quality numbers measured on synthetic data mean nothing for production.

Domain: elevator fault analysis (assets O21/O22/O23), Vietnamese.
Run: ``python -m training.common.synthetic --out-dir training/data``
or ``make gen-synthetic-data``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from training.common.schemas import (
    ChatMessage,
    EmbeddingExample,
    SFTExample,
    ToolCall,
    ToolUseExample,
    write_jsonl,
)

N_PER_TRACK = 20

_ASSETS = ["O21", "O22", "O23"]

_FAULTS = [
    ("cửa tầng", "cảm biến cửa", "F-101"),
    ("động cơ kéo", "quá nhiệt động cơ", "F-204"),
    ("cáp tải", "mòn cáp", "F-310"),
    ("phanh", "má phanh mòn", "F-402"),
    ("bộ điều khiển", "lỗi bo mạch", "F-515"),
]

_DOCS = {
    "cửa tầng": (
        "Hướng dẫn kiểm tra cảm biến cửa tầng: đo điện trở tiếp điểm, "
        "vệ sinh ray cửa, kiểm tra lò xo đóng cửa."
    ),
    "động cơ kéo": (
        "Quy trình xử lý quá nhiệt động cơ kéo: kiểm tra quạt làm mát, "
        "đo dòng điện pha, kiểm tra dầu bôi trơn hộp số."
    ),
    "cáp tải": (
        "Tiêu chuẩn thay cáp tải: đo đường kính cáp, kiểm tra sợi đứt, "
        "thay khi mòn vượt 10% đường kính."
    ),
    "phanh": (
        "Bảo trì phanh: đo khe hở má phanh, kiểm tra lò xo phanh, "
        "thử phanh khẩn cấp."
    ),
    "bộ điều khiển": (
        "Chẩn đoán bo mạch điều khiển: đọc mã lỗi, kiểm tra nguồn cấp, "
        "thay tụ lọc nếu phồng."
    ),
}


def _split(i: int) -> str:
    """Deterministic 70/15/15 split by index."""
    if i % 20 < 14:
        return "train"
    if i % 20 < 17:
        return "val"
    return "test"


def generate_embedding(n: int = N_PER_TRACK, seed: int = 7) -> list[EmbeddingExample]:
    """Contrastive retrieval examples: query + correct doc + near misses."""
    rng = random.Random(seed)
    examples: list[EmbeddingExample] = []
    for i in range(n):
        asset = _ASSETS[i % len(_ASSETS)]
        fault, cause, code = _FAULTS[i % len(_FAULTS)]
        query = f"Thang máy {asset} bị lỗi {fault} ({code}) thì kiểm tra như thế nào?"
        positive = _DOCS[fault]
        others = [f for f in _FAULTS if f[0] != fault]
        rng.shuffle(others)
        negatives = [_DOCS[o[0]] for o in others[:2]]
        examples.append(
            EmbeddingExample(
                id=f"emb-{i + 1:03d}",
                query=query,
                positive=positive,
                negatives=negatives,
                source="synthetic",
                split=_split(i),
            )
        )
    return examples


def _sft_messages(asset: str, fault: str, cause: str, code: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "Bạn là fault_investigator. Phân tích lỗi thang máy và chỉ "
                "kết luận dựa trên evidence thu thập được."
            ),
        ),
        ChatMessage(
            role="user",
            content=f"Phân tích nguyên nhân các lỗi của thang máy {asset} trong tháng 10/2026.",
        ),
        ChatMessage(
            role="assistant",
            content=f"Tôi sẽ tra lịch sử lỗi của {asset} trước.",
            tool_calls=[
                ToolCall(
                    name="search_fault_history",
                    arguments={
                        "asset_id": asset,
                        "start_date": "2026-10-01",
                        "end_date": "2026-10-31",
                    },
                )
            ],
        ),
        ChatMessage(
            role="tool",
            content=f"Lịch sử lỗi {asset}: {code} {cause} lặp lại nhiều lần.",
            tool_call_id="call-1",
        ),
        ChatMessage(
            role="assistant",
            content=f"Tôi tra thêm tài liệu kỹ thuật về {fault}.",
            tool_calls=[
                ToolCall(
                    name="search_technical_documents",
                    arguments={"query": f"{fault} thang máy {code}", "limit": 5},
                )
            ],
        ),
        ChatMessage(
            role="tool",
            content=f"Tài liệu: {_DOCS[fault]}",
            tool_call_id="call-2",
        ),
        ChatMessage(
            role="assistant",
            content=(
                f"Nguyên nhân chính: {cause} ({code}) trên thang máy {asset}. "
                f"Bằng chứng: lịch sử lỗi lặp lại + tài liệu {fault}. "
                "Khuyến nghị: kiểm tra theo quy trình tài liệu và lên lịch bảo trì."
            ),
        ),
    ]


def generate_sft(n: int = N_PER_TRACK, seed: int = 7) -> list[SFTExample]:
    """SFT examples: mission request + tool calls + grounded conclusion."""
    examples: list[SFTExample] = []
    for i in range(n):
        asset = _ASSETS[i % len(_ASSETS)]
        fault, cause, code = _FAULTS[i % len(_FAULTS)]
        examples.append(
            SFTExample(
                id=f"sft-{i + 1:03d}",
                mission_id="fault_analysis",
                messages=_sft_messages(asset, fault, cause, code),
                source="synthetic",
                split=_split(i),
            )
        )
    return examples


_TOOLS = [
    "search_fault_history",
    "get_maintenance_records",
    "search_technical_documents",
    "query_asset_relationships",
    "get_sensor_data",
    "calculate_failure_rate",
]


def generate_tool_use(n: int = N_PER_TRACK, seed: int = 7) -> list[ToolUseExample]:
    """Tool-use examples: context + expected tool call (incl. traps).

    The last 3 cases are traps: the request is outside the mission, so
    the correct behaviour is to call NO tool (expected_tool=None).
    """
    examples: list[ToolUseExample] = []
    for i in range(n):
        asset = _ASSETS[i % len(_ASSETS)]
        fault, cause, code = _FAULTS[i % len(_FAULTS)]
        if i >= n - 3:
            traps = [
                (
                    "Đặt lịch họp với khách hàng về hợp đồng thuê văn phòng.",
                    None,
                    {},
                ),
                (
                    "Tính lương tháng 10 cho kỹ thuật viên bảo trì.",
                    None,
                    {},
                ),
                (
                    "Viết bài quảng cáo cho dịch vụ cho thuê văn phòng.",
                    None,
                    {},
                ),
            ]
            context, tool, args = traps[i - (n - 3)]
            examples.append(
                ToolUseExample(
                    id=f"tool-{i + 1:03d}",
                    mission_id="fault_analysis",
                    context=context,
                    available_tools=list(_TOOLS),
                    expected_tool=tool,
                    expected_args=args,
                    source="synthetic",
                    split=_split(i),
                )
            )
            continue
        cases = [
            (
                f"Tra lịch sử lỗi của thang máy {asset} trong tháng 10/2026.",
                "search_fault_history",
                {"asset_id": asset, "start_date": "2026-10-01", "end_date": "2026-10-31"},
            ),
            (
                f"Lấy hồ sơ bảo trì của thang máy {asset}.",
                "get_maintenance_records",
                {"asset_id": asset},
            ),
            (
                f"Tìm tài liệu kỹ thuật về {fault} ({code}).",
                "search_technical_documents",
                {"query": f"{fault} thang máy {code}", "limit": 5},
            ),
            (
                f"Tính tỉ lệ lỗi của thang máy {asset} trong tháng 10/2026.",
                "calculate_failure_rate",
                {"asset_id": asset, "start_date": "2026-10-01", "end_date": "2026-10-31"},
            ),
        ]
        context, tool, args = cases[i % len(cases)]
        examples.append(
            ToolUseExample(
                id=f"tool-{i + 1:03d}",
                mission_id="fault_analysis",
                context=context,
                available_tools=list(_TOOLS),
                expected_tool=tool,
                expected_args=args,
                source="synthetic",
                split=_split(i),
            )
        )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic training datasets.")
    parser.add_argument("--out-dir", default="training/data", help="Output directory.")
    parser.add_argument("--n", type=int, default=N_PER_TRACK, help="Examples per track.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    counts = {
        "synthetic_embedding.jsonl": write_jsonl(
            out_dir / "synthetic_embedding.jsonl", generate_embedding(args.n)
        ),
        "synthetic_sft.jsonl": write_jsonl(
            out_dir / "synthetic_sft.jsonl", generate_sft(args.n)
        ),
        "synthetic_tool_use.jsonl": write_jsonl(
            out_dir / "synthetic_tool_use.jsonl", generate_tool_use(args.n)
        ),
    }
    for name, count in counts.items():
        print(f"wrote {out_dir / name}: {count} examples (source=synthetic)")
    print("NOTE: synthetic data is for pipeline testing only — never use it")
    print("as evidence of model quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
