class FunctionLibrary:
    def __init__(self):
        self.library_terms = [
            # Special term
            "f",
            # Degree 1
            "x",
            "y",
            # Degree 2
            "tf.pow(x, 2)",
            "tf.multiply(x, y)",
            "tf.pow(y, 2)",
            # Degree 3
            "tf.pow(x, 3)",
            "tf.multiply(tf.pow(x, 2), y)",
            "tf.multiply(x, tf.pow(y, 2))",
            "tf.pow(y, 3)",
            # Degree 4
            "tf.pow(x, 4)",
            "tf.multiply(tf.pow(x, 3), y)",
            "tf.multiply(tf.pow(x, 2), tf.pow(y, 2))",
            "tf.multiply(x, tf.pow(y, 3))",
            "tf.pow(y, 4)",
            # Degree 5
            "tf.pow(x, 5)",
            "tf.multiply(tf.pow(x, 4), y)",
            "tf.multiply(tf.pow(x, 3), tf.pow(y, 2))",
            "tf.multiply(tf.pow(x, 2), tf.pow(y, 3))",
            "tf.multiply(x, tf.pow(y, 4))",
            "tf.pow(y, 5)",
        ]
        self.terms_number = len(self.library_terms)

    def build_functions(self, coefficients):
        """Build the functions for acceleration.
        Args:
            lambda (list): List of coefficients for x.
        Returns:
            function (str): Function for x.
        """
        function_x = ""
        for i in range(self.terms_number):
            term = self.library_terms[i]
            if coefficients[i] != 0:
                function_x += f"+cx{i}*{term}"
        function_x = function_x[1:]
        return function_x

    def get_functions(self, coefficients):
        function_x = ""
        for i in range(self.terms_number):
            term = self.library_terms[i]
            if coefficients[i] != 0:
                function_x += f"+{coefficients[i].numpy():.4f}*{term}"
        function_x = function_x[1:].replace("+-", "-")
        return function_x
