#Francesca Craievich IN0501042

#eccezione per stack vuoto
class EmptyStackException(Exception):
    def __init__(self, message="Empty stack"):
        self.message = message
        super().__init__(self.message) 
          
class Stack:
    def __init__(self):
        self.data = []

    def push(self, x):
        self.data.append(x)

    def pop(self):
        if self.data == []:
            raise EmptyStackException
        res = self.data[-1]
        self.data = self.data[0:-1]
        return res

    def __str__(self):
        return " ".join([str(s) for s in self.data])
 
#classe per costruire e valutare espressioni matematiche
class Expression:
    def __init__(self):
        raise NotImplementedError
        
#metodo di classe che divide l'espressione nei vari token e li stiva nello stack a seconda se sono operandi o operatori (usando un dizionario dispatch)

    @classmethod
    def from_program(cls, text, dispatch):
        tokens = text.split() 
        stack = Stack()  

        for token in tokens:
            if token not in dispatch:  
                try:
                    value = float(token)
                    stack.push(Constant(value)) 
                except ValueError:
                    var_name = str(token)
                    stack.push(Variable(var_name))             
                         
            else:                
                arity = dispatch[token].arity    
                param = []
                for i in range(arity):
                    param.append(stack.pop())
                stack.push(dispatch[token](param))
                    
        result = stack.pop()
        return result

    def evaluate(self, env):
        raise NotImplementedError()

class MissingVariableException(Exception):
    def __init__(self, message="Variable is missing."):
        self.message = message
        super().__init__(self.message)

#classe che prende un dizionario env che contiene le variabili e le valuta, incrementa il valore in env e le rappresenta con __str__

class Variable(Expression):
    def __init__(self, name):
        self.name = str(name)

    def evaluate(self, env):  
        if self.name in env:
            return env[self.name]  
        else:
            raise MissingVariableException  
   
    def increment(self, env, val): 
        if self.name in env:
            env[self.name] += val
        else:
            raise MissingVariableException
                
    def __str__(self):
        return self.name
        
#classe per le costanti   
class Constant(Expression):
    def __init__(self, value):
        self.value = value

    def evaluate(self, env):
        return self.value

    def __str__(self):
        return str(self.value)

#operazione con valutazione e rappresentazione in stringa 
class Operation(Expression):
    def __init__(self, args):
        self.args = args

    def evaluate(self, env):
        values= []
        for arg in self.args:                      
            values.append(arg.evaluate(env))           
        return self.op(*values)                   
                                   
    def op(self, *args):
        raise NotImplementedError()

    def __str__(self):  
        res = f'({self.name} '
        for arg in self.args:
            res += f'{arg} '
        res += f')'
        return res
    
#1 argomento -> operazioni unarie
class UnaryOperation(Operation):
    arity = 1  

#2 argomenti -> operazioni binarie
class BinaryOperation(Operation):
    arity = 2

class Add(BinaryOperation):
    name = "+"

    def op(self, x, y):
        return x + y

class Subtract(BinaryOperation):
    name = "-"

    def op(self, x, y):
        return x - y

class Divide(BinaryOperation):
    name = "/"  

    def op(self, x, y):
        if y == 0:
            raise ZeroDivisionError("Error: divided by zero")
        return x / y

class Multiply(BinaryOperation):
    name = "*"

    def op(self, x, y):
        return x * y   
    
class Power(BinaryOperation):
    name = "**"

    def op(self, x, y):
        return x ** y
   
class Modulus(BinaryOperation):
    name = "%"

    def op(self, x, y):
        return x % y

class Reciprocal(UnaryOperation):
    name = "1/"

    def op(self, x):
        if x == 0:
            raise ZeroDivisionError("Error: divided by zero")
        return 1 / x

class AbsoluteValue(UnaryOperation):
    name = "abs"

    def op(self, x):
        return abs(x)

class GreaterThan(BinaryOperation):
    name = ">"

    def op(self, x, y):
        return x > y

class GreaterThanOrEqual(BinaryOperation):
    name = ">="

    def op(self, x, y):
        return x >= y

class LessThan(BinaryOperation):
    name = "<"

    def op(self, x, y):
        return x < y

class LessThanOrEqual(BinaryOperation):
    name = "<="

    def op(self, x, y):
        return x <= y

class Equal(BinaryOperation):
    name = "=="

    def op(self, x, y):
        return x == y

class NotEqual(BinaryOperation):
    name = "!="

    def op(self, x, y):
        return x != y

#var alloc -> allocare variabili per renderle disponibili nel codice, valore di default zero

class VarAlloc(UnaryOperation):       
    name = "alloc"

    def __init__(self, args):
        self.args = args

    def evaluate(self, env):
        env[f'{self.args[0]}'] = 0 


#n var valloc -> variabile e n (risultato espressione che ritorna un intero) che è la lunghezza dell'array

class NVarAlloc(BinaryOperation):
    name = "valloc"

    def __init__(self, args):
        self.args = args
        self.var = args[0]
        self.n = args[1]

    def evaluate(self, env):
        n = self.n.evaluate(env)
        if isinstance(n, float) and not n.is_integer():
            raise ValueError("The size must be an integer")
        env[f'{self.var}'] = [0] * int(n)   
        
#expr x setq -> imposta valore variabile x al valore di expr e ritorna il nuovo valore di x

class SetQ(BinaryOperation):
    name = "setq"
    arity = 2

    def __init__(self, args):
        self.args = args
        self.x = args[0]  
        self.expr = args[1]

    def evaluate(self, env):
        if f'{self.x}' in env:
            env[f'{self.x}'] = self.expr.evaluate(env)
           
            return env[f'{self.x}']  
        else:
            raise MissingVariableException

#expr n x setv -> imposta indice n dell'array con valore di expr. ritorna nuovo valore di x in posizione n

class SetV(Operation):
    name = "setv"
    arity = 3

    def __init__(self, args):
        self.args = args
        self.x = args[0]
        self.n = args[1]
        self.expr = args[2]

    def evaluate(self, env):
        
        n = self.n.evaluate(env)  
        if isinstance(n, float) and not n.is_integer():
            raise ValueError('Index must be an integer')
        
        n = int(n)

        if f'{self.x}' not in env:
            raise MissingVariableException
        
        list_len = len(env[f'{self.x}'])
        if n > list_len:
            raise IndexError('Index must be in the list')

        env[f'{self.x}'][n] = self.expr.evaluate(env)          
        return env[f'{self.x}'][n]  

#sequenze 
class Prog(Operation):

    def __init__(self, args):
        self.args = args   

    def evaluate(self, env):
        res = []
        for expr in self.args:
            res.append(expr.evaluate(env))

        return res[-1]   

class BinaryProg(Prog):
    name = "prog2"
    arity = 2


class TernaryProg(Prog):
    name = "prog3"
    arity = 3

class QuaternaryProg(Prog):
    name = "prog4"
    arity = 4

#if-no if-yes cond -> se cond torna true viene valutata if-yes, sennò viene valutata if-no

class If(Operation):
    name = "if"
    arity = 3

    def __init__(self, args):
        self.args = args
        self.cond = args[0]   
        self.if_yes = args[1] 
        self.if_no = args[2]  

    def evaluate(self, env):
        if self.cond.evaluate(env):      
            return self.if_yes.evaluate(env)
        
        return self.if_no.evaluate(env)  

#expr cond while -> valuta cond e se è vera valuta expr finchè cond diventa falsa 

class While(BinaryOperation):
    name = "while"
    arity = 2

    def __init__(self, args):
        self.args = args
        self.cond = args[0]  
        self.expr = args[1]  
        
    def evaluate(self, env):
        if not isinstance(self.cond, Expression):
            raise TypeError("Condition must be an istance of Expression")
        
        while self.cond.evaluate(env):
            self.expr.evaluate(env)

        return    

#expr end start i for -> valuta expr con variabile di iterazione 'i' da start a end-1 con incrementi di 1
#alloco la variabile di iterazione e le assegno il valore di start
 
class For(Operation):
    name = "for"
    arity = 4

    def __init__(self, args):
        self.args = args
        self.i = args[0]     
        self.start = args[1] 
        self.end = args[2]   
        self.expr = args[3]  
        
    def evaluate(self, env):
        
        if not self.start.evaluate(env).is_integer():
            raise ValueError("Start must be an integer")
        
        if not self.end.evaluate(env).is_integer():
            raise ValueError("End must be an integer")
        
        VarAlloc([self.i.name]).evaluate(env)  
        counter = Variable(self.i)
        end = self.end.evaluate(env)

        SetQ([f'{self.i}', self.start]).evaluate(env)  

        while(counter.evaluate(env) < end):
            self.expr.evaluate(env)  
            counter.increment(env, 1)

        return
    

#expr f defsub -> associo alla varibile f una espressione che verrà poi chiamata da call

class DefSub(BinaryOperation):
    name = "defsub"
    arity = 2

    def __init__(self, args):
        self.args = args
        self.f = args[0]  
        self.expr = args[1]  

    def evaluate(self, env):
        env[f'{self.f}'] = self.expr

#f call -> valuta expr associata a f
class Call(UnaryOperation):
    name = "call"
    arity = 1

    def __init__(self, args):
        self.args = args
        self.f = args[0] 

    def evaluate(self, env):
        if f"{self.f}" not in env:
            raise MissingVariableException
        
        env[f"{self.f}"].evaluate(env)  

#expr print -> valuta expr e stampa il risultato. restituisce risultato di expr

class Print(UnaryOperation):
    name = "print"
    arity = 1

    def __init__(self, args):
        self.args = args
        self.expr = args[0]  

    def evaluate(self, env):
        res = self.expr.evaluate(env)
        print(f'Result: {res}')
        return res  
 
#non esegue nessuna operazione
class Nop(Operation):
    name = "nop"
    arity = 0

    def __init__(self, args):
        self.args = args

    def evaluate(self, env):
        pass
   
#dizionario per le operazioni 
d={
    "+": Add,
    "-": Subtract,
    "*": Multiply,
    "/": Divide,
    "**": Power,
    "%": Modulus,
    "1/": Reciprocal,
    "abs": AbsoluteValue,
    ">": GreaterThan,
    ">=": GreaterThanOrEqual,
    "<": LessThan,
    "<=": LessThanOrEqual,
    "==": Equal,
    "!=": NotEqual,
    "alloc": VarAlloc,
    "valloc": NVarAlloc,
    "setq": SetQ,
    "setv": SetV,
    "prog2": BinaryProg,
    "prog3": TernaryProg,
    "prog4": QuaternaryProg,
    "if-else": If,
    "while": While,
    "for": For,
    "defsub": DefSub,
    "call": Call,
    "print": Print,
    "nop": Nop
}


example = "2 3 + x * 6 5 - / abs 2 ** y 1/ + 1/"
e = Expression.from_program(example, d)
print(e)
res = e.evaluate({"x": 3, "y": 7})
print(res)


