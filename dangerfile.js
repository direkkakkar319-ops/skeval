const { danger, warn, fail } = require('danger');

// Rule 1: Warn if PR has no description
if (danger.github.pr.body == null || danger.github.pr.body.length === 0) {
  warn('Please provide a description for this PR.');
}

// Rule 2: Warn if PR is too large (>500 lines)
const bigPRThreshold = 500;
if (danger.github.pr.additions + danger.github.pr.deletions > bigPRThreshold) {
  warn(`This PR is quite large (${danger.github.pr.additions + danger.github.pr.deletions} lines). Consider splitting it into smaller PRs to make review easier.`);
}

// Rule 3: Missing Tests Warning (src changes but no tests changes)
const hasSrcChanges = danger.git.modified_files.some(path => path.includes('src/'));
const hasTestChanges = danger.git.modified_files.some(path => path.includes('tests/'));
if (hasSrcChanges && !hasTestChanges) {
  warn('Files in the `src/` directory were modified, but no corresponding changes were made in the `tests/` directory. Please ensure tests are updated.');
}
