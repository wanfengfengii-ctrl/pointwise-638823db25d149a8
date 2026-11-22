import json
import os
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    suite = unittest.defaultTestLoader.discover(
        os.path.join(ROOT, "tests"), top_level_dir=ROOT)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    from block_examples import baseline_comparison, block_verification
    summary = {"baseline": baseline_comparison(), "blocks": block_verification()}
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
