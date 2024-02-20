turingQuote= "A computer would deserve to be called intelligent if it could deceive a human into believing that it was human"

turingQuote= turingQuote.upper()
print(turingQuote)
characters = len(turingQuote) 
words=len(turingQuote.split())
print(f"this has {words} words this has character{characters} ")

val='Hi'
name= 'folks'
print(f"{val} {name} {turingQuote}")
print(f"the Turing quote contains {characters} characters and {words} words")

x=0
numbers=[7, 0, 14, -3, 81]
for integer in numbers:
    print(f"The Integer in position {x} is {integer}")
    x+=1


