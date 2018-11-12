def append_bug(test_append=[]):
    print 'test_append is: ', test_append
    test_append.append(-3), test_append.append(-4)


if __name__ == "__main__":
    print 'the 1st call'
    append_bug()
    print 'the 2nd call'
    append_bug()
    print 'the 3rd call'
    append_bug()
