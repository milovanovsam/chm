# Repo for computational tasks.

## Task list:
1. Term 1:
    * `S1T0_types_convertions`
    * `S1T1_compute_function`
    * `S1T2_solve_linear_system`
    * `S1T3_newton_method`
1. Term 2:
    * `S2T1_optimization`
    * `S2T2_interpolation`
    * `S2T3_approximation`
1. Term 3:
    * `S3T1_integration`
    * `S3T2_solve_ode`

Each task should be solved in Python>=3.6, look corresponding folders for code examples and test scripts.
You should write the code that solves the problem and is verified by test script. Check readme.md files in subfolders for details (in Russian!).

## Common Python code structure:
* `<TASK_NAME>/`
    * `__init__.py`
    * `py/`
        * `__init__.py`
        * `test_<subtask_name>.py` - test script that checks your solutions. You have several options to run it:
          * `python -m pytest path/to/test_<task_name>.py`
          * `pytest path/to/test_<task_name>.py`
          * set pytest as PyCharm default test runner and press `Run test` at `def test_<task_name>():` row.
        * `<subtask_name>.py` - dummy code, you should replace it with yours.

## Contacts
Feel free to ask:
* sagoyewatha@mail.ru
* [Term 1] https://t.me/chm_1sem
* [Term 3] https://t.me/chm_3sem
