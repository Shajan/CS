from behave import given, when, then

@given('an adder')
def _(context): # The function name here is irrelavant
    pass

@when('adding {a:d} and {b:d}') # ':d' for integer, default is str
def _(context, a: int, b: int):
    context.sum = a + b

@then('we get {s:d}')
def _(context, s: int):
    assert context.sum == s
