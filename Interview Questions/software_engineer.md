# Software Engineering Interview Questions

## Basic Algorithm

### Q: What is a stack? What are the two basic operations of a stack?
A stack is a linear data structure with three basic operations: push (insertion of an element to the stack from the top), pop (removal of the latest element added to the stack). Some implementations of stack also allow peek, a function enabling you to see an element in a stack without modifying it. Stacks use a last-in, first-out structure – so the last element added to the stack is the first element that can be removed. Queues are a similar data structure, which work with a first-in, first-out structure. Stacks are usually implemented with an array or a linked list. You might be asked to implement a stack in an interview and to implement different operations.

## Coding Test

### Q: Reverse an Integer
```python
def reverse(x):
    x = str(x)
    if x[0] == '-':
        return int('-' + x[:0:-1])
    else:
        return int(x[::-1])
```