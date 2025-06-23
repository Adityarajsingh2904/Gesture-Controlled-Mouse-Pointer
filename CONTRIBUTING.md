# Contributing

Thank you for considering contributing to this project!

## Setup Instructions

1. Clone the repository and install the dependencies:

```bash
git clone <repo-url>
cd Gesture-Controlled-Mouse-Pointer
pip install -r requirements.txt
```

2. Ensure you are using Python 3.8 or higher.

## Branch Naming Rules

Use short descriptive branch names prefixed with your initials, for example:

```
jd-fix-scroll-bug
```

## How to Test Changes

Run a basic syntax check over all Python files:

```bash
python -m py_compile $(git ls-files '*.py')
```

If you add tests, execute them with `pytest`.

## Reporting Bugs

Please open an issue with detailed steps to reproduce the problem. Include any error messages and your operating system information.
