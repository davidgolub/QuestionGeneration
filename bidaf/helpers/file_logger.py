class FileLogger(object):
    """ Simple logger to insert stuff into a file """
    def __init__(self, path):
        self.file = open(path, 'w')

    def write(self, text, print_text=True):
        if print_text:
            print("FILE LOGGER: %s" % text)
        self.file.write(str(text) + "\n")
        self.file.flush() 