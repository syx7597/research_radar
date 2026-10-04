"""CLI for explicit read-only structured radar record queries."""
import argparse
import json
from pathlib import Path

from .adapter import MODES, RadarRecordAdapter
from scripts import audit_radar_review_workflow as workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="JSON object; exact entity and attribute required")
    parser.add_argument("--mode", choices=MODES, default="accepted-independent")
    parser.add_argument("--workflow", type=Path, default=workflow.DEFAULT_WORKFLOW)
    args = parser.parse_args()
    try:
        result = RadarRecordAdapter(args.workflow).query(json.loads(args.query), args.mode)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, json.dumps(dict(status="validation_or_query_error", error=str(exc)), ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
