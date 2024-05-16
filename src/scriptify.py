# from colorama import Fore, Style

# def print_scriptify():
#     print(Fore.GREEN + Style.BRIGHT, end="")
#     pattern = [
#         "  SSSS   CCCC  RRRR   IIII  PPPP   TTTTTT  IIIII  FFFFFF   Y   Y ",
#         "  S      C   C R   R   I    P   P    T       I    F         Y Y  ",
#         "   SSSS  C     RRRR    I    PPPP     T       I    FFFFF      Y   ",
#         "      S  C     R   R   I    P        T       I    F          Y   ",
#         "  SSSS   CCCC  R   R  IIII  P        T     IIIII  F          Y   "
#     ]
    
#     for line in pattern:
#         print(line)

#     print(Style.RESET_ALL, end="")
from colorama import Fore, Style
import time

def print_scriptify():
    print(Fore.GREEN + Style.BRIGHT, end="")
    pattern = [
        "  SSSS   CCCC  RRRR   IIII  PPPP   TTTTTT  IIIII  FFFFFF   Y   Y ",
        "  S      C   C R   R   I    P   P    T       I    F         Y Y  ",
        "   SSSS  C     RRRR    I    PPPP     T       I    FFFFF      Y   ",
        "      S  C     R   R   I    P        T       I    F          Y   ",
        "  SSSS   CCCC  R   R  IIII  P        T     IIIII  F          Y   "
    ]

    for line in pattern:
        print(line)
        time.sleep(0.25)  # Add a delay of 0.2 seconds between printing each line

    print(Style.RESET_ALL, end="")

# print_scriptify()
