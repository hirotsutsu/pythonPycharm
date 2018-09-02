class Dog:
    name = ""
    def bark(self):
        m = self.name + ": Bow-wow!"
        print m

pochi = Dog()
pochi.name = "Pochi"
pochi.bark()