class param(object):
    def __init__(self, data):
        self.line_thickness = int(data['common_args']['line_thickness'])
        self.same = float(data['common_args']['same'])
        self.w1 = int(data['tools']['polygon']['w1'])
        self.h1 = int(data['tools']['polygon']['h1'])
        self.w2 = int(data['tools']['polygon']['w2'])
        self.h2 = int(data['tools']['polygon']['h2'])
        self.w3 = int(data['tools']['polygon']['w3'])
        self.h3 = int(data['tools']['polygon']['h3'])
        self.w4 = int(data['tools']['polygon']['w4'])
        self.h4 = int(data['tools']['polygon']['h4'])

    def line_thickness(self):
        return self.line_thickness

    def same(self):
        return self.same

    def w1(self):
        return self.w1

    def h1(self):
        return self.h1

    def w2(self):
        return self.w2

    def h2(self):
        return self.h2

    def w3(self):
        return self.w3

    def h3(self):
        return self.h3

    def w4(self):
        return self.w4

    def h4(self):
        return self.h4