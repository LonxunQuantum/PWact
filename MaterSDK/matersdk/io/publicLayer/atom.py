from prettytable import PrettyTable


class Atom(object):
    def __init__(self,
                atomic_number:int,
                coordination:list,
                magnetic_moment:float=0
                ):
        self.atomic_number = atomic_number
        self.coordination = coordination
        self.magnetic_moment = magnetic_moment

    def __repr__(self):
        table = PrettyTable(["Element", "Coordination", "Magnetic moment"])
        table.add_row([self.atomic_number, self.coordination, self.magnetic_moment])
        print(table)
        return ""

    def __str__(self):
        return self.__repr__()
