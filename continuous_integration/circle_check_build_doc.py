"""Check whether we or not we should build the documentation

If the last commit message has a "[skip: doc]" marker, do not build
the doc.

We always build the documentation for jobs that are not related to a specific
PR (e.g. a merge to master or a maintenance branch).

If this is a PR, check that if there are some files in this PR that are under
the "doc/" or "examples/" folders, otherwise skip.

"""
import sys
import os
from subprocess import check_output

pr_url = os.environ.get('CI_PULL_REQUEST')
if not pr_url:
    # The documentation should be always built when executed from one of the
    # main branches
    print("OK: not a pull request")
    sys.exit(0)

compare_url = os.environ.get('CIRCLE_COMPARE_URL')
if not compare_url:
    print("SKIP: undefined CIRCLE_COMPARE_URL variable")
    sys.exit(0)

git_range = compare_url.rsplit('/', 1)[-1]
if '...' not in git_range:
    print("SKIP: invalid git range: %s" % git_range)
    sys.exit(0)

filenames = check_output("git diff --name-only".split() + [git_range])
filenames = filenames.decode('utf-8').split()
for filename in filenames:
    if filename.startswith(u'doc/') or filename.startswith(u'examples/'):
        print("OK: detected doc impacting file at %s" % filename)
        sys.exit(0)

print("SKIP: no doc impacting files detected:")
print(u"\n".join(filenames))
