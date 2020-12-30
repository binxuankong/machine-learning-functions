# Python Interview Questions

## Basic

### Q: Is Python case sensitive?
Yes, Python is a case sensitive language.

### Q: Is indentation required in Python?
Yes, indentation is necessary for Python. It specifies a block of code. All code within loops, classes, functions, etc is specified within an indented block. It is usually done using tab (2 or 4 space characters). If a Python code is not indented necessarily, it will not execute accurately and will throw errors.

### Q: What is the difference between list and tuple in Python?
Lists and Tuples are both sequence data types that can store a collection of objects in Python. The objects stored in both sequences can have different data types.
| List | Tuple |
| ---- | ----- |
| List is multable (can be edited) | Tuple is immutable (cannot be edited) |
| List is slower than tuple | Tuple is faster than list |
| Syntax: `list1 = [1, 2, 3]` | Syntax: `tup1 = (1, 2, 3)` |

### Q: What is the difference between list and array in Python?
Arrays and lists in Python have the same way of storing data. Arrays in Python can only contain elements of same data types i.e., data type of array should be homogeneous. It is a thin wrapper around C language arrays and consumes far less memory than lists.
Lists in Python can contain elements of different data types i.e., data type of lists can be heterogeneous. It has the disadvantage of consuming large memory.

### Q: What are negative indexes and how are they used?
Python sequences can be index in positive and negative numbers. For positive index, 0 is the first index, 1 is the second index and so forth. For negative index, (-1) is the last index and (-2) is the second last index and so forth.

### Q: What is the difference between break, continue and pass in Python?
| Break | Allows loop termination when some condition is met and the control is transferred to the next statement (outside of the loop). |
| Continue | Allows skipping some part of a loop when some specific condition is met and the control is transferred to the beginning of the loop. |
| Pass | Allows skipping the execution of some block of code. This is basically a null operation, nothing happens when this is executed. |

### Q: What is slicing in Python?
Syntax for slicing is `[start : stop : step]`
- `start` is the starting index from where to slice a list or tuple
- `stop` is the ending index or where to stop.
- `step` is the number of steps to jump.
Default value for `start` is 0, `stop` is number of items, `step` is 1. Slicing can be done on strings, arrays, lists, and tuples.


## Intermediate

### Q: How would you setup many projects where each one uses different versions of Python and third party libraries?
Using virtual environments `virtualenv`.

### Q: How to fetch every third item in a list?
```python
# Using slicing
lst[2::3]

# Using for loop
[x for i, x in enumerate(lst) if (i+1) % 3 == 0]
```

### Q: What is a lambda function?
Lambda function is an anonymous function. This function can have any number of arguments, but can have just one expression. It is generally used in situations requiring an anonymous function for a short time period.
Example:
```python
a = lambda x, y: x + y
print(a(5,6))
```

### Q: Explain list comprehensions and how they are used in Python.
List comprehensions provide a concise way to create lists. A list is traditionally created using square brackests. But with a list comprehenstion, these brackets contain an expression followed by a `for` clause and then `if` clauses, when necessary. Evaluation the given expression in the context of these for and if clauses produces a list.
```python
old_list = [1, 0, -2, 4, -3]
new_list = [x**2 for x in old_list if x > 0]
print(new_list)
```

### Q: What are `*args` and `**kwargs` in Python? How do we use it?
We use `*args` when we are not sure how many arguments are going to be passed to a function, or if we want to pass a stored list or tuple of arguments to a function. `**kwargs` is used when we don't know how many keyword arguments will be passed to a function, or it can be used to pass the values of a dictionary as keyword arguments. The identifiers *args* and *kwargs* are convention, we could use any identifiers such as `*bob` or `**billy` but that is not recommended.


## Web Scraping

### Q: How to scrape data from IMDb top 250 movies page, extracting only the movie name, year and rating?
```python
import sys
import requests
from bs4 import BeautifulSoup

url = '<a href="http://www.imdb.com/chart/top">http://www.imdb.com/chart/top</a>'
response - requests.get(url)
soup = BeautifulSoup(response.text)
tr = soup.findChildren('tr')
tr = iter(tr)
next(tr)

for movie in tr:
    title = movie.find('td', {'class': 'titleColumn'}).find('a').contents[0]
    year = movie.find('td', {'class': 'titleCoulmn'}).find('span', {'class': 'secondaryInfo'}).contents[0]
    rating = movie.find('td', {'class': 'ratingColumn imdbRating'}).find('strong').contents[0]
    row = title + ' - ' + year + ' - ' + rating
    print(row)
```


## Data Analysis

### Q: What are the most commonly used data analysis libraries in Python?
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciKit

### Q: What is the difference between NumPy and SciPy?
1. NumPy mostly focuses on array data type and the most basic operations: indexing, sorting, reshaping, basic elementwise functions, and so on.
2. SciPy mostly focuses on numerical code.
3. One of NumPy's important goal is to be compatible, so it tries to retain all features supported by either of its predecessors. So NumPy contains some linear algebra function.
4. SciPy contains more fully-featured versions of the linear algebra modules, as well as many other numerical algorithms.

### Q: What is pandas?
Pandas is a Python open-source library that provides high-performance and flexible data structures and data analysis tools that make working with relational or labeled data both easy and intuitive.