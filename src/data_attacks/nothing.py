class Al:
    def __init__(self, num=False) -> None:
        self.num = num
        pass

    def __call__(self, net, img, cl):
        return img, 0
