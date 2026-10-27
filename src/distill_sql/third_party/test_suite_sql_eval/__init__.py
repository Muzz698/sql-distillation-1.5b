"""Vendored from https://github.com/taoyds/test-suite-sql-eval (commit e97acc546ecbee8fa27fa8dbf025ef61493a876c).

Apache License 2.0 (see LICENSE). Used for Spider's official exact-set-match metric and
difficulty ("hardness") labels. Changes are marked `PATCH (distill-sql)`:
  * process_sql.py: nltk.word_tokenize -> an equivalent regex (no nltk data download)
  * evaluation.py: relative import; the execution-match module is not vendored
"""
