from mypackage.code import do_something

def test_something_func():
    assert do_something() == 15
    assert do_something(6) == 18
    assert do_something(0) == 0