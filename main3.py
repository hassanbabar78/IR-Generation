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

# ============================ MIPS ASSEMBLER ============================

class MIPSAssembler:
    def __init__(self):
        self.data_section = []
        self.text_section = []
        self.current_temp = 0
        self.label_counter = 0
        self.var_addresses = {}
        self.temp_addresses = {}
        self.next_address = 0
        self.string_constants = {}
        self.next_string_label = 0
    
    def generate_mips(self, tac_instructions: List[TACInstruction]) -> str:
        """Generate MIPS assembly from TAC instructions"""
        self.data_section = [".data"]
        self.text_section = [".text", ".globl main"]
        
        # First pass: collect variables and allocate memory
        self._collect_variables(tac_instructions)
        
        # Second pass: generate MIPS code
        self._generate_code(tac_instructions)
        
        return "\n".join(self.data_section + [""] + self.text_section)
    
    def _collect_variables(self, tac_instructions: List[TACInstruction]):
        """Collect all variables and allocate memory for them"""
        for instr in tac_instructions:
            # Collect variables from instructions
            if instr.result and instr.result.startswith(('x', 'y', 'z', 'a', 'b', 'c', 'i', 'j', 'k', 'result', 'sum', 'flag')):
                if instr.result not in self.var_addresses:
                    self.var_addresses[instr.result] = f"var_{instr.result}"
                    self.data_section.append(f"var_{instr.result}: .word 0")
            
            # Collect temporaries
            for arg in [instr.arg1, instr.arg2, instr.result]:
                if arg and arg.startswith('t') and arg not in self.temp_addresses:
                    self.temp_addresses[arg] = f"temp_{arg}"
                    self.data_section.append(f"temp_{arg}: .word 0")
    
    def _generate_code(self, tac_instructions: List[TACInstruction]):
        """Generate MIPS assembly code from TAC"""
        for instr in tac_instructions:
            if instr.op == TACOp.LABEL:
                self.text_section.append(f"{instr.arg1}:")
            
            elif instr.op == TACOp.ASSIGN:
                self._generate_assign(instr)
            
            elif instr.op in [TACOp.ADD, TACOp.SUB, TACOp.MUL, TACOp.DIV]:
                self._generate_arithmetic(instr)
            
            elif instr.op in [TACOp.EQ, TACOp.NE, TACOp.LT, TACOp.LE, TACOp.GT, TACOp.GE]:
                self._generate_comparison(instr)
            
            elif instr.op in [TACOp.AND, TACOp.OR]:
                self._generate_logical(instr)
            
            elif instr.op == TACOp.NEG:
                self._generate_negation(instr)
            
            elif instr.op == TACOp.NOT:
                self._generate_logical_not(instr)
            
            elif instr.op == TACOp.GOTO:
                self.text_section.append(f"j {instr.arg1}")
            
            elif instr.op == TACOp.IF_FALSE:
                self._generate_conditional_jump(instr)
            
            elif instr.op == TACOp.PARAM:
                # Simple parameter passing - in real compiler would use stack
                reg = self._get_next_param_reg()
                self._load_value(instr.arg1, '$t0')
                self.text_section.append(f"move {reg}, $t0")
            
            elif instr.op == TACOp.CALL:
                self._generate_function_call(instr)
            
            elif instr.op == TACOp.RETURN:
                self._generate_return(instr)
    
    def _generate_assign(self, instr: TACInstruction):
        """Generate assignment: result = arg1"""
        if instr.arg1.isdigit() or (instr.arg1[0] == '-' and instr.arg1[1:].isdigit()):
            # Immediate value
            self.text_section.append(f"li $t0, {instr.arg1}")
        else:
            # Variable to variable
            self._load_value(instr.arg1, '$t0')
        
        self._store_value(instr.result, '$t0')
    
    def _generate_arithmetic(self, instr: TACInstruction):
        """Generate arithmetic operations"""
        self._load_value(instr.arg1, '$t0')
        self._load_value(instr.arg2, '$t1')
        
        op_map = {
            TACOp.ADD: "add",
            TACOp.SUB: "sub", 
            TACOp.MUL: "mul",
            TACOp.DIV: "div"
        }
        
        mips_op = op_map.get(instr.op)
        if mips_op:
            self.text_section.append(f"{mips_op} $t2, $t0, $t1")
            self._store_value(instr.result, '$t2')
    
    def _generate_comparison(self, instr: TACInstruction):
        """Generate comparison operations"""
        self._load_value(instr.arg1, '$t0')
        self._load_value(instr.arg2, '$t1')
        
        # Set $t2 to 1 if condition true, 0 otherwise
        op_map = {
            TACOp.EQ: "seq",
            TACOp.NE: "sne",
            TACOp.LT: "slt",
            TACOp.LE: "sle", 
            TACOp.GT: "sgt",
            TACOp.GE: "sge"
        }
        
        mips_op = op_map.get(instr.op)
        if mips_op:
            self.text_section.append(f"{mips_op} $t2, $t0, $t1")
            self._store_value(instr.result, '$t2')
    
    def _generate_logical(self, instr: TACInstruction):
        """Generate logical operations"""
        self._load_value(instr.arg1, '$t0')
        self._load_value(instr.arg2, '$t1')
        
        if instr.op == TACOp.AND:
            self.text_section.append("and $t2, $t0, $t1")
        elif instr.op == TACOp.OR:
            self.text_section.append("or $t2, $t0, $t1")
        
        self._store_value(instr.result, '$t2')
    
    def _generate_negation(self, instr: TACInstruction):
        """Generate negation operation"""
        self._load_value(instr.arg1, '$t0')
        self.text_section.append("neg $t1, $t0")
        self._store_value(instr.result, '$t1')
    
    def _generate_logical_not(self, instr: TACInstruction):
        """Generate logical NOT operation"""
        self._load_value(instr.arg1, '$t0')
        self.text_section.append("xori $t1, $t0, 1")  # Flip 0<->1
        self._store_value(instr.result, '$t1')
    
    def _generate_conditional_jump(self, instr: TACInstruction):
        """Generate conditional jump: ifFalse arg1 goto arg2"""
        self._load_value(instr.arg1, '$t0')
        self.text_section.append(f"beqz $t0, {instr.arg2}")
    
    def _generate_function_call(self, instr: TACInstruction):
        """Generate function call"""
        self.text_section.append(f"jal func_{instr.arg1}")
        # Store return value
        self.text_section.append("move $t0, $v0")
        self._store_value(instr.result, '$t0')
    
    def _generate_return(self, instr: TACInstruction):
        """Generate return statement"""
        if instr.arg1:
            self._load_value(instr.arg1, '$v0')
        self.text_section.append("jr $ra")
    
    def _load_value(self, source: str, dest_reg: str):
        """Load value from variable or temporary into register"""
        if source.isdigit() or (source[0] == '-' and source[1:].isdigit()):
            self.text_section.append(f"li {dest_reg}, {source}")
        elif source in self.var_addresses:
            self.text_section.append(f"lw {dest_reg}, {self.var_addresses[source]}")
        elif source in self.temp_addresses:
            self.text_section.append(f"lw {dest_reg}, {self.temp_addresses[source]}")
        else:
            # Assume it's a variable we haven't seen yet
            if source not in self.var_addresses:
                self.var_addresses[source] = f"var_{source}"
                self.data_section.append(f"var_{source}: .word 0")
            self.text_section.append(f"lw {dest_reg}, var_{source}")
    
    def _store_value(self, dest: str, src_reg: str):
        """Store value from register to variable or temporary"""
        if dest in self.var_addresses:
            self.text_section.append(f"sw {src_reg}, {self.var_addresses[dest]}")
        elif dest in self.temp_addresses:
            self.text_section.append(f"sw {src_reg}, {self.temp_addresses[dest]}")
        else:
            # Assume it's a variable we haven't seen yet
            if dest not in self.var_addresses:
                self.var_addresses[dest] = f"var_{dest}"
                self.data_section.append(f"var_{dest}: .word 0")
            self.text_section.append(f"sw {src_reg}, var_{dest}")
    
    def _get_next_param_reg(self):
        """Get next parameter register (simplified)"""
        regs = ['$a0', '$a1', '$a2', '$a3']
        return regs[self.current_temp % len(regs)]

# ============================ COMPLETE COMPILER ============================

class CompleteCompiler:
    def __init__(self):
        self.ir_generator = IRGenerator()
        self.mips_assembler = MIPSAssembler()
    
    def compile(self, ast: Dict[str, Any], output_file: str = None):
        try:
            # Generate TAC
            print("Generating Intermediate Representation...")
            tac_instructions = self.ir_generator.generate_ir(ast)
            
            print("\nThree Address Code:\n")
            self.ir_generator.print_ir()
            
            # Generate MIPS assembly
            print("\nGenerated MIPS Assembly\n:")
            mips_code = self.mips_assembler.generate_mips(tac_instructions)
            print(mips_code)
            
            if output_file:
                with open(f"{output_file}.s", "w") as f:
                    f.write(mips_code)
                print(f"\nMIPS assembly written to {output_file}.s")
            
            return True
            
        except CompilerError as e:
            print(f"Compilation Error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            return False

# Test function
def test_complete_compiler():
    # Test case 1: Simple arithmetic and return
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
                        'name': 'a',
                        'data_type': 'int',
                        'value': {'type': 'literal', 'value': 15}
                    },
                    {
                        'type': 'variable_declaration',
                        'name': 'b',
                        'data_type': 'int', 
                        'value': {'type': 'literal', 'value': 3}
                    },
                    {
                        'type': 'return_statement',
                        'argument': {
                            'type': 'binary_expression',
                            'operator': '*',
                            'left': {'type': 'identifier', 'name': 'a'},
                            'right': {'type': 'identifier', 'name': 'b'}
                        }
                    }
                ]
            }
        ]
    }

    compiler = CompleteCompiler()
    compiler.compile(sample_ast, "output")

if __name__ == "__main__":
    test_complete_compiler()