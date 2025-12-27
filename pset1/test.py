class A:
    answer = 42
    def init(self):
        self.answer = 21
        self.add = lambda x, y: x.answer + y
    def add(self, y):
        return self.answer - y
print(A() + 5)