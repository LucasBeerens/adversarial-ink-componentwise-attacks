class Al:
    def __init__(self, num=False) -> None:
        """
        Initialize the Al class. This is does nothing to the image.
        This 'attack' is only used for visualisation purposes. With
        this attack it is possible to pass a list of attacks including
        this one that does nothing, so we can compare attacks against
        the original

        Args:
        - num (bool): Whether to use numerical approximation.
        """
        self.num = num
        pass

    def __call__(self, net, img, cl):
        """
        Perform the attack algorithm.

        Args:
        - net: The neural network model.
        - img: The input image.
        - cl: The target class.

        Returns:
        - The original image and an epsilon value of 0.
        """
        return img, 0
