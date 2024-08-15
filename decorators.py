import os
import sys
import traceback

# defining a error decorator
def error_traceback(func, *args, **kwargs):

    # inner1 is a Wrapper function in 
    # which the argument is called
    
    # inner function can access the outer local
    # functions like in this case "func"
    def inner1(*args, **kwargs):
        try:
            out = func(*args, **kwargs)
            return out
        except Exception as e:
            print("ERROR:")
            traceback.print_exc()
           
    return inner1