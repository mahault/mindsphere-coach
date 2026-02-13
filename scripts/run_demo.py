"""
Quick demo script: run MindSphere Coach locally.

Usage:
    python scripts/run_demo.py
"""

import uvicorn


def main():
    print("=" * 60)
    print("  MindSphere Coach — Interactive ToM Coaching Agent")
    print("=" * 60)
    print()
    print("Starting server at http://localhost:8000")
    print("Open your browser to begin the coaching session.")
    print()
    print("API docs: http://localhost:8000/docs")
    print("Press Ctrl+C to stop.")
    print()

    uvicorn.run(
        "mindsphere.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
