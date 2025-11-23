import enum
from typing import List, Optional, Union, Dict, Any

class IRType(enum.Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    VOID = "void"

class TACOp(enum.Enum):
    ASSIGN = "="
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    NEG = "neg"
    NOT = "!"
    AND = "&&"
    OR = "||"
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    GOTO = "goto"
    IF_TRUE = "if_true"
    IF_FALSE = "if_false"
    PARAM = "param"
    CALL = "call"
    RETURN = "return"
    LABEL = "label"

class TACInstruction:
    def __init__(self, op: TACOp, arg1: str = None, arg2: str = None, result: str = None):
        self.op = op
        self.arg1 = arg1
        self.arg2 = arg2
        self.result = result
        self.label = None
    
    def set_label(self, label: str):
        self.label = label
  
    def __str__(self):
        parts = []
        if self.label:
            parts.append(f"{self.label}:")
        
        if self.op == TACOp.LABEL:
            return f"{self.arg1}:"
        elif self.op == TACOp.GOTO:
            return f"goto {self.arg1}"
        elif self.op == TACOp.IF_TRUE:
            return f"if {self.arg1} goto {self.arg2}"
        elif self.op == TACOp.IF_FALSE:
            return f"ifFalse {self.arg1} goto {self.arg2}"
        elif self.op == TACOp.PARAM:
            return f"param {self.arg1}"
        elif self.op == TACOp.CALL:
            if self.result:
                return f"{self.result} = call {self.arg1}, {self.arg2}"
            else:
                return f"call {self.arg1}, {self.arg2}"
        elif self.op == TACOp.RETURN:
            if self.arg1:
                return f"return {self.arg1}"
            else:
                return "return"
        elif self.op == TACOp.ASSIGN:
            return f"{self.result} = {self.arg1}"
        elif self.arg2 is not None:
            return f"{self.result} = {self.arg1} {self.op.value} {self.arg2}"
        elif self.arg1 is not None:
            return f"{self.result} = {self.op.value} {self.arg1}"
        else:
            return f"{self.result} = {self.op.value}"

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.functions = {}
        self.temp_counter = 0
        self.label_counter = 0
    
    def add_symbol(self, name: str, type: IRType, scope: str = "global") -> bool:
        """Add symbol to table, returns False if already exists in current scope"""
        key = f"{scope}.{name}" if scope != "global" else name
        if key in self.symbols:
            return False
        self.symbols[key] = {
            'type': type,
            'scope': scope,
            'address': f"var_{name}"
        }
        return True
    
    def get_symbol(self, name: str, scope: str = None) -> Optional[Dict]:
        """Get symbol from current scope or global scope"""
        # Check current scope first
        if scope:
            key = f"{scope}.{name}"
            if key in self.symbols:
                return self.symbols[key]
        
        # Check global scope
        if name in self.symbols:
            return self.symbols[name]
        
        return None
    
    def add_function(self, name: str, return_type: IRType, parameters: List[Dict]):
        """Add function to function table"""
        self.functions[name] = {
            'return_type': return_type,
            'parameters': parameters,
            'param_types': [self._map_type(p['data_type']) for p in parameters]
        }
    
    def get_function(self, name: str) -> Optional[Dict]:
        return self.functions.get(name)
    
    def new_temp(self) -> str:
        temp = f"t{self.temp_counter}"
        self.temp_counter += 1
        return temp
    
    def new_label(self) -> str:
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label
    
    def _map_type(self, type_str: str) -> IRType:
        type_map = {
            'int': IRType.INT,
            'float': IRType.FLOAT,
            'bool': IRType.BOOL,
            'void': IRType.VOID
        }
        return type_map.get(type_str, IRType.INT)

class CompilerError(Exception):
    def __init__(self, message: str, line: int = None):
        self.message = message
        self.line = line
        super().__init__(f"Error at line {line}: {message}" if line else f"Error: {message}")

class TypeChecker:
    @staticmethod
    def is_numeric_type(type: IRType) -> bool:
        return type in [IRType.INT, IRType.FLOAT]
    
    @staticmethod
    def is_boolean_type(type: IRType) -> bool:
        return type == IRType.BOOL
    
    @staticmethod
    def get_expression_type(node: Dict, symbol_table: SymbolTable, current_scope: str) -> IRType:
        """Determine the type of an expression"""
        if node['type'] == 'literal':
            value = node['value']
            if isinstance(value, bool):
                return IRType.BOOL
            elif isinstance(value, int):
                return IRType.INT
            elif isinstance(value, float):
                return IRType.FLOAT
            else:
                return IRType.INT  # Default
        
        elif node['type'] == 'identifier':
            symbol = symbol_table.get_symbol(node['name'], current_scope)
            if not symbol:
                raise CompilerError(f"Undeclared variable: {node['name']}")
            return symbol['type']
        
        elif node['type'] == 'binary_expression':
            left_type = TypeChecker.get_expression_type(node['left'], symbol_table, current_scope)
            right_type = TypeChecker.get_expression_type(node['right'], symbol_table, current_scope)
            
            op = node['operator']
            
            # Arithmetic operators require numeric types
            if op in ['+', '-', '*', '/', '%']:
                if not (TypeChecker.is_numeric_type(left_type) and TypeChecker.is_numeric_type(right_type)):
                    raise CompilerError(f"Arithmetic operation '{op}' requires numeric types, got {left_type.value} and {right_type.value}")
                # For mixed types, promote to float
                if left_type == IRType.FLOAT or right_type == IRType.FLOAT:
                    return IRType.FLOAT
                return IRType.INT
            
            # Comparison operators require compatible types
            elif op in ['==', '!=', '<', '<=', '>', '>=']:
                if left_type != right_type:
                    raise CompilerError(f"Comparison operation '{op}' requires compatible types, got {left_type.value} and {right_type.value}")
                return IRType.BOOL
            
            # Logical operators require boolean types
            elif op in ['&&', '||']:
                if not (TypeChecker.is_boolean_type(left_type) and TypeChecker.is_boolean_type(right_type)):
                    raise CompilerError(f"Logical operation '{op}' requires boolean types, got {left_type.value} and {right_type.value}")
                return IRType.BOOL
            
        elif node['type'] == 'unary_expression':
            operand_type = TypeChecker.get_expression_type(node['operand'], symbol_table, current_scope)
            op = node['operator']
            
            if op == '-':
                if not TypeChecker.is_numeric_type(operand_type):
                    raise CompilerError(f"Unary minus requires numeric type, got {operand_type.value}")
                return operand_type
            elif op == '!':
                if not TypeChecker.is_boolean_type(operand_type):
                    raise CompilerError(f"Logical NOT requires boolean type, got {operand_type.value}")
                return IRType.BOOL
        
        elif node['type'] == 'function_call':
            func = symbol_table.get_function(node['name'])
            if not func:
                raise CompilerError(f"Undefined function: {node['name']}")
            return func['return_type']
        
        return IRType.INT  # Default fallback

class IRGenerator:
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.instructions: List[TACInstruction] = []
        self.current_scope = "global"
        self.current_function = None
        self.type_checker = TypeChecker()
    
    def generate_ir(self, ast: Dict[str, Any]) -> List[TACInstruction]:
        """Generate TAC from AST"""
        try:
            self._validate_ast_structure(ast)
            
            if ast['type'] == 'program':
                # First pass: collect function declarations
                for node in ast['body']:
                    if node['type'] == 'function_declaration':
                        self._collect_function_declaration(node)
                
                # Second pass: generate code
                for node in ast['body']:
                    self._generate_node(node)
            return self.instructions
        except KeyError as e:
            raise CompilerError(f"Missing required field in AST: {e}")
        except Exception as e:
            raise CompilerError(f"IR generation failed: {str(e)}")
    
    def _validate_ast_structure(self, ast: Dict):
        """Validate AST structure has required fields"""
        if 'type' not in ast:
            raise CompilerError("AST node missing 'type' field")
        
        if ast['type'] == 'program':
            if 'body' not in ast:
                raise CompilerError("Program node missing 'body' field")
        elif ast['type'] == 'function_declaration':
            required = ['name', 'return_type', 'body']
            for field in required:
                if field not in ast:
                    raise CompilerError(f"Function declaration missing '{field}' field")
        elif ast['type'] == 'variable_declaration':
            required = ['name', 'data_type']
            for field in required:
                if field not in ast:
                    raise CompilerError(f"Variable declaration missing '{field}' field")
    
    def _collect_function_declaration(self, node: Dict[str, Any]):
        """Collect function information (first pass)"""
        func_name = node['name']
        return_type = self._map_type(node['return_type'])
        parameters = node.get('parameters', [])
        
        self.symbol_table.add_function(func_name, return_type, parameters)
    
    def _generate_node(self, node: Dict[str, Any]):
        node_type = node['type']
        
        if node_type == 'variable_declaration':
            self._generate_variable_declaration(node)
        elif node_type == 'function_declaration':
            self._generate_function_declaration(node)
        elif node_type == 'assignment':
            self._generate_assignment(node)
        elif node_type == 'binary_expression':
            return self._generate_binary_expression(node)
        elif node_type == 'unary_expression':
            return self._generate_unary_expression(node)
        elif node_type == 'identifier':
            return self._generate_identifier(node)
        elif node_type == 'literal':
            return self._generate_literal(node)
        elif node_type == 'if_statement':
            self._generate_if_statement(node)
        elif node_type == 'while_statement':
            self._generate_while_statement(node)
        elif node_type == 'return_statement':
            self._generate_return_statement(node)
        elif node_type == 'function_call':
            return self._generate_function_call(node)
        elif node_type == 'expression_statement':
            self._generate_expression_statement(node)
        else:
            raise CompilerError(f"Unsupported node type: {node_type}")
    
    def _generate_variable_declaration(self, node: Dict[str, Any]):
        var_name = node['name']
        var_type = self._map_type(node['data_type'])
        
        # Check for duplicate declaration in current scope
        if not self.symbol_table.add_symbol(var_name, var_type, self.current_scope):
            raise CompilerError(f"Duplicate variable declaration: {var_name}")
        
        if 'value' in node and node['value']:
            # Type check the initialization value
            init_type = self.type_checker.get_expression_type(node['value'], self.symbol_table, self.current_scope)
            if init_type != var_type:
                raise CompilerError(f"Type mismatch in variable '{var_name}': expected {var_type.value}, got {init_type.value}")
            
            temp = self._generate_node(node['value'])
            self.instructions.append(TACInstruction(TACOp.ASSIGN, temp, None, var_name))
    
    def _generate_function_declaration(self, node: Dict[str, Any]):
        func_name = node['name']
        return_type = self._map_type(node['return_type'])
        
        # Function label
        func_label = f"func_{func_name}"
        self.instructions.append(TACInstruction(TACOp.LABEL, func_label))
        
        # Save current context
        old_scope = self.current_scope
        old_function = self.current_function
        self.current_scope = func_name
        self.current_function = func_name
        
        # Parameters
        for param in node.get('parameters', []):
            param_type = self._map_type(param['data_type'])
            if not self.symbol_table.add_symbol(param['name'], param_type, self.current_scope):
                raise CompilerError(f"Duplicate parameter name: {param['name']}")
        
        # Function body
        for stmt in node['body']:
            self._generate_node(stmt)
        
        # Add implicit return if needed
        if return_type != IRType.VOID and not self._has_return_statement(node['body']):
            self.instructions.append(TACInstruction(TACOp.RETURN, None))
        
        # Restore context
        self.current_scope = old_scope
        self.current_function = old_function
    
    def _generate_assignment(self, node: Dict[str, Any]):
        var_name = node['left']['name']
        
        # Check if variable exists
        symbol = self.symbol_table.get_symbol(var_name, self.current_scope)
        if not symbol:
            raise CompilerError(f"Assignment to undeclared variable: {var_name}")
        
        var_type = symbol['type']
        
        # Type check the assignment
        right_type = self.type_checker.get_expression_type(node['right'], self.symbol_table, self.current_scope)
        if var_type != right_type:
            raise CompilerError(f"Type mismatch in assignment to '{var_name}': expected {var_type.value}, got {right_type.value}")
        
        temp = self._generate_node(node['right'])
        self.instructions.append(TACInstruction(TACOp.ASSIGN, temp, None, var_name))
        return var_name
    
    def _generate_binary_expression(self, node: Dict[str, Any]) -> str:
        # Type check the expression
        expr_type = self.type_checker.get_expression_type(node, self.symbol_table, self.current_scope)
        
        left_temp = self._generate_node(node['left'])
        right_temp = self._generate_node(node['right'])
        result_temp = self.symbol_table.new_temp()
        
        op_map = {
            '+': TACOp.ADD,
            '-': TACOp.SUB,
            '*': TACOp.MUL,
            '/': TACOp.DIV,
            '%': TACOp.MOD,
            '==': TACOp.EQ,
            '!=': TACOp.NE,
            '<': TACOp.LT,
            '<=': TACOp.LE,
            '>': TACOp.GT,
            '>=': TACOp.GE,
            '&&': TACOp.AND,
            '||': TACOp.OR
        }
        
        op = op_map.get(node['operator'])
        if not op:
            raise CompilerError(f"Unsupported operator: {node['operator']}")
        
        self.instructions.append(TACInstruction(op, left_temp, right_temp, result_temp))
        return result_temp
    
    def _generate_unary_expression(self, node: Dict[str, Any]) -> str:
        # Type check the expression
        expr_type = self.type_checker.get_expression_type(node, self.symbol_table, self.current_scope)
        
        operand_temp = self._generate_node(node['operand'])
        result_temp = self.symbol_table.new_temp()
        
        if node['operator'] == '-':
            self.instructions.append(TACInstruction(TACOp.NEG, operand_temp, None, result_temp))
        elif node['operator'] == '!':
            self.instructions.append(TACInstruction(TACOp.NOT, operand_temp, None, result_temp))
        else:
            raise CompilerError(f"Unsupported unary operator: {node['operator']}")
        
        return result_temp
    
    def _generate_identifier(self, node: Dict[str, Any]) -> str:
        var_name = node['name']
        symbol = self.symbol_table.get_symbol(var_name, self.current_scope)
        if not symbol:
            raise CompilerError(f"Undeclared variable: {var_name}")
        return var_name
    
    def _generate_literal(self, node: Dict[str, Any]) -> str:
        temp = self.symbol_table.new_temp()
        self.instructions.append(TACInstruction(TACOp.ASSIGN, str(node['value']), None, temp))
        return temp
    
    def _generate_if_statement(self, node: Dict[str, Any]):
        # Type check condition (must be boolean)
        cond_type = self.type_checker.get_expression_type(node['condition'], self.symbol_table, self.current_scope)
        if not self.type_checker.is_boolean_type(cond_type):
            raise CompilerError(f"If condition must be boolean, got {cond_type.value}")
        
        condition_temp = self._generate_node(node['condition'])
        
        false_label = self.symbol_table.new_label()
        end_label = self.symbol_table.new_label()
        
        # Jump to else if false
        self.instructions.append(TACInstruction(TACOp.IF_FALSE, condition_temp, false_label))
        
        # Then branch
        for stmt in node['consequent']:
            self._generate_node(stmt)
        
        # Jump to end
        self.instructions.append(TACInstruction(TACOp.GOTO, end_label))
        
        # Else branch
        self.instructions.append(TACInstruction(TACOp.LABEL, false_label))
        if 'alternate' in node:
            for stmt in node['alternate']:
                self._generate_node(stmt)
        
        # End label
        self.instructions.append(TACInstruction(TACOp.LABEL, end_label))
    
    def _generate_while_statement(self, node: Dict[str, Any]):
        # Type check condition (must be boolean)
        cond_type = self.type_checker.get_expression_type(node['condition'], self.symbol_table, self.current_scope)
        if not self.type_checker.is_boolean_type(cond_type):
            raise CompilerError(f"While condition must be boolean, got {cond_type.value}")
        
        start_label = self.symbol_table.new_label()
        end_label = self.symbol_table.new_label()
        
        # Start of loop
        self.instructions.append(TACInstruction(TACOp.LABEL, start_label))
        
        # Condition
        condition_temp = self._generate_node(node['condition'])
        self.instructions.append(TACInstruction(TACOp.IF_FALSE, condition_temp, end_label))
        
        # Loop body
        for stmt in node['body']:
            self._generate_node(stmt)
        
        # Jump back to condition
        self.instructions.append(TACInstruction(TACOp.GOTO, start_label))
        
        # End of loop
        self.instructions.append(TACInstruction(TACOp.LABEL, end_label))
    
    def _generate_return_statement(self, node: Dict[str, Any]):
        if 'argument' in node and node['argument']:
            # Type check return value matches function return type
            return_type = self.type_checker.get_expression_type(node['argument'], self.symbol_table, self.current_scope)
            func = self.symbol_table.get_function(self.current_function)
            if func and func['return_type'] != return_type:
                raise CompilerError(f"Return type mismatch: expected {func['return_type'].value}, got {return_type.value}")
            
            return_temp = self._generate_node(node['argument'])
            self.instructions.append(TACInstruction(TACOp.RETURN, return_temp))
        else:
            # Check if function expects void return
            func = self.symbol_table.get_function(self.current_function)
            if func and func['return_type'] != IRType.VOID:
                raise CompilerError(f"Function {self.current_function} expects return value of type {func['return_type'].value}")
            self.instructions.append(TACInstruction(TACOp.RETURN, None))
    
    def _generate_function_call(self, node: Dict[str, Any]) -> str:
        func_name = node['name']
        func = self.symbol_table.get_function(func_name)
        
        if not func:
            raise CompilerError(f"Undefined function: {func_name}")
        
        # Check argument count
        provided_args = node.get('arguments', [])
        expected_args = func['parameters']
        
        if len(provided_args) != len(expected_args):
            raise CompilerError(f"Function {func_name} expects {len(expected_args)} arguments, got {len(provided_args)}")
        
        # Check argument types
        for i, (arg, expected_param) in enumerate(zip(provided_args, expected_args)):
            arg_type = self.type_checker.get_expression_type(arg, self.symbol_table, self.current_scope)
            expected_type = self._map_type(expected_param['data_type'])
            if arg_type != expected_type:
                raise CompilerError(f"Argument {i+1} type mismatch in call to {func_name}: expected {expected_type.value}, got {arg_type.value}")
        
        # Push parameters
        for arg in provided_args:
            arg_temp = self._generate_node(arg)
            self.instructions.append(TACInstruction(TACOp.PARAM, arg_temp))
        
        # Function call
        result_temp = self.symbol_table.new_temp()
        self.instructions.append(TACInstruction(TACOp.CALL, func_name, str(len(provided_args)), result_temp))
        
        return result_temp
    
    def _generate_expression_statement(self, node: Dict[str, Any]):
        self._generate_node(node['expression'])
    
    def _map_type(self, type_str: str) -> IRType:
        type_map = {
            'int': IRType.INT,
            'float': IRType.FLOAT,
            'bool': IRType.BOOL,
            'void': IRType.VOID
        }
        return type_map.get(type_str, IRType.INT)
    
    def _has_return_statement(self, body: List[Dict]) -> bool:
        for node in body:
            if node['type'] == 'return_statement':
                return True
            if node['type'] in ['if_statement', 'while_statement']:
                if self._check_nested_returns(node):
                    return True
        return False
    
    def _check_nested_returns(self, node: Dict) -> bool:
        if node['type'] == 'if_statement':
            has_return = any(self._has_return_statement([stmt]) for stmt in node['consequent'])
            if 'alternate' in node:
                has_return = has_return and any(self._has_return_statement([stmt]) for stmt in node['alternate'])
            return has_return
        elif node['type'] == 'while_statement':
            return any(self._has_return_statement([stmt]) for stmt in node['body'])
        return False
    
    def print_ir(self):
        """Print the generated TAC instructions"""
        for i, instr in enumerate(self.instructions):
            print(f"{i:3d}: {instr}")

# Test function
def test_ir_generator():
    #  Test case
    #  int x = 10; int y = 5; if (x > y) { return 1; } else { return 0; }
    sample_ast = {
        'type': 'program',
        'body': [
            {
                'type': 'function_declaration',
                'name': 'main',
                'return_type': 'int',
                'parameters': [],
                'body': [
                    {
                        'type': 'variable_declaration',
                        'name': 'x',
                        'data_type': 'int',
                        'value': {'type': 'literal', 'value': 10}
                    },
                    {
                        'type': 'variable_declaration',
                        'name': 'y', 
                        'data_type': 'int',
                        'value': {'type': 'literal', 'value': 5}
                    },
                    {
                        'type': 'if_statement',
                        'condition': {
                            'type': 'binary_expression',
                            'operator': '>',
                            'left': {'type': 'identifier', 'name': 'x'},
                            'right': {'type': 'identifier', 'name': 'y'}
                        },
                        'consequent': [
                            {
                                'type': 'return_statement',
                                'argument': {'type': 'literal', 'value': 1}
                            }
                        ],
                        'alternate': [
                            {
                                'type': 'return_statement',
                                'argument': {'type': 'literal', 'value': 0}
                            }
                        ]
                    }
                ]
            }
        ]
    }
            
    generator = IRGenerator()
    try:
        instructions = generator.generate_ir(sample_ast)
        print("Generated TAC Instructions:")
        print("=" * 40)
        generator.print_ir()
        return instructions
    except CompilerError as e:
        print(f"Compilation error: {e}")

if __name__ == "__main__":
    test_ir_generator()