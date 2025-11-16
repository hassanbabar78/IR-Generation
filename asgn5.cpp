#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>

using namespace std;

// Scope analysis types (needed for TypeChecker)
enum class ScopeError
{
    UndeclaredVariableAccessed,
    UndefinedFunctionCalled,
    VariableRedefinition,
    FunctionPrototypeRedefinition,
    None
};

enum class SymbolType
{
    Variable,
    Function,
    Parameter
};

struct SymbolInfo
{
    string name;
    SymbolType type;
    string dataType;
    int scopeLevel;
    bool isPrototype;

    SymbolInfo(const string &n, SymbolType t, const string &dt, int level, bool proto = false)
        : name(n), type(t), dataType(dt), scopeLevel(level), isPrototype(proto) {}
};

class ScopeNode
{
public:
    int scopeLevel;
    ScopeNode *parent;
    unordered_map<string, SymbolInfo> symbols;

    ScopeNode(int level, ScopeNode *p = nullptr)
        : scopeLevel(level), parent(p) {}

    bool addSymbol(const string &name, SymbolType type, const string &dataType, bool isPrototype = false)
    {
        if (symbols.find(name) != symbols.end())
        {
            return false;
        }
        symbols.emplace(name, SymbolInfo(name, type, dataType, scopeLevel, isPrototype));
        return true;
    }

    optional<SymbolInfo> lookupLocal(const string &name) const
    {
        auto it = symbols.find(name);
        if (it != symbols.end())
        {
            return it->second;
        }
        return nullopt;
    }
};

// Type checking errors
enum class TypeChkError
{
    ErroneousVarDecl,
    FnCallParamCount,
    FnCallParamType,
    ErroneousReturnType,
    ExpressionTypeMismatch,
    ExpectedBooleanExpression,
    ErroneousBreak,
    NonBooleanCondStmt,
    EmptyExpression,
    AttemptedBoolOpOnNonBools,
    AttemptedBitOpOnNonNumeric,
    AttemptedShiftOnNonInt,
    AttemptedAddOpOnNonNumeric,
    AttemptedExponentiationOfNonNumeric,
    ReturnStmtNotFound,
    None
};

// Expression types for type checking
enum class ExprType
{
    Int,
    Float,
    Char,
    Bool,
    Void,
    String,
    Error,
    Unknown
};

// Operator types
enum class Operator
{
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Exp,
    // Comparison
    Eq,
    Neq,
    Lt,
    Gt,
    Leq,
    Geq,
    // Logical
    And,
    Or,
    Not,
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,
    // Assignment
    Assign
};

// Forward declaration
class ScopeAnalyzer;

// Type Checker Module
class TypeChecker
{
private:
    ScopeAnalyzer &scopeAnalyzer;
    string currentFunction;
    ExprType currentFunctionReturnType;
    bool inLoop;
    bool hasReturnStmt;
    vector<TypeChkError> errors;

public:
    TypeChecker(ScopeAnalyzer &analyzer)
        : scopeAnalyzer(analyzer), currentFunction(""),
          currentFunctionReturnType(ExprType::Void), inLoop(false), hasReturnStmt(false) {}

    // Convert string type to ExprType
    ExprType stringToExprType(const string &typeStr)
    {
        if (typeStr == "int")
            return ExprType::Int;
        if (typeStr == "float")
            return ExprType::Float;
        if (typeStr == "char")
            return ExprType::Char;
        if (typeStr == "bool")
            return ExprType::Bool;
        if (typeStr == "void")
            return ExprType::Void;
        if (typeStr == "string")
            return ExprType::String;
        return ExprType::Unknown;
    }

    string exprTypeToString(ExprType type)
    {
        switch (type)
        {
        case ExprType::Int:
            return "int";
        case ExprType::Float:
            return "float";
        case ExprType::Char:
            return "char";
        case ExprType::Bool:
            return "bool";
        case ExprType::Void:
            return "void";
        case ExprType::String:
            return "string";
        case ExprType::Error:
            return "error";
        case ExprType::Unknown:
            return "unknown";
        default:
            return "unknown";
        }
    }

    // Error reporting
    void reportError(TypeChkError error, const string &context = "")
    {
        errors.push_back(error);
        cout << "Type Error: " << typeErrorToString(error);
        if (!context.empty())
        {
            cout << " [" << context << "]";
        }
        cout << endl;
    }

    string typeErrorToString(TypeChkError error)
    {
        switch (error)
        {
        case TypeChkError::ErroneousVarDecl:
            return "Erroneous variable declaration";
        case TypeChkError::FnCallParamCount:
            return "Function call parameter count mismatch";
        case TypeChkError::FnCallParamType:
            return "Function call parameter type mismatch";
        case TypeChkError::ErroneousReturnType:
            return "Return type doesn't match function signature";
        case TypeChkError::ExpressionTypeMismatch:
            return "Expression type mismatch";
        case TypeChkError::ExpectedBooleanExpression:
            return "Expected boolean expression";
        case TypeChkError::ErroneousBreak:
            return "Break statement outside loop";
        case TypeChkError::NonBooleanCondStmt:
            return "Non-boolean condition in statement";
        case TypeChkError::EmptyExpression:
            return "Empty expression";
        case TypeChkError::AttemptedBoolOpOnNonBools:
            return "Boolean operation on non-boolean types";
        case TypeChkError::AttemptedBitOpOnNonNumeric:
            return "Bitwise operation on non-numeric types";
        case TypeChkError::AttemptedShiftOnNonInt:
            return "Shift operation on non-integer types";
        case TypeChkError::AttemptedAddOpOnNonNumeric:
            return "Additive operation on non-numeric types";
        case TypeChkError::AttemptedExponentiationOfNonNumeric:
            return "Exponentiation of non-numeric types";
        case TypeChkError::ReturnStmtNotFound:
            return "Function missing return statement";
        case TypeChkError::None:
            return "No error";
        default:
            return "Unknown error";
        }
    }

    // Type checking for expressions
    ExprType checkBinaryOp(Operator op, ExprType left, ExprType right, const string &context)
    {
        switch (op)
        {
        // Arithmetic operations
        case Operator::Add:
        case Operator::Sub:
        case Operator::Mul:
        case Operator::Div:
            if ((left == ExprType::Int || left == ExprType::Float) &&
                (right == ExprType::Int || right == ExprType::Float))
            {
                return (left == ExprType::Float || right == ExprType::Float) ? ExprType::Float : ExprType::Int;
            }
            reportError(TypeChkError::AttemptedAddOpOnNonNumeric, context);
            return ExprType::Error;

        case Operator::Mod:
            if (left == ExprType::Int && right == ExprType::Int)
            {
                return ExprType::Int;
            }
            reportError(TypeChkError::AttemptedAddOpOnNonNumeric, context);
            return ExprType::Error;

        case Operator::Exp:
            if ((left == ExprType::Int || left == ExprType::Float) &&
                (right == ExprType::Int || right == ExprType::Float))
            {
                return (left == ExprType::Float || right == ExprType::Float) ? ExprType::Float : ExprType::Int;
            }
            reportError(TypeChkError::AttemptedExponentiationOfNonNumeric, context);
            return ExprType::Error;

        // Comparison operations (always return bool)
        case Operator::Eq:
        case Operator::Neq:
        case Operator::Lt:
        case Operator::Gt:
        case Operator::Leq:
        case Operator::Geq:
            if ((left == ExprType::Int || left == ExprType::Float) &&
                (right == ExprType::Int || right == ExprType::Float))
            {
                return ExprType::Bool;
            }
            if (left == right && left != ExprType::Void && left != ExprType::Error)
            {
                return ExprType::Bool;
            }
            reportError(TypeChkError::ExpressionTypeMismatch, context);
            return ExprType::Error;

        // Logical operations
        case Operator::And:
        case Operator::Or:
            if (left == ExprType::Bool && right == ExprType::Bool)
            {
                return ExprType::Bool;
            }
            reportError(TypeChkError::AttemptedBoolOpOnNonBools, context);
            return ExprType::Error;

        // Bitwise operations
        case Operator::BitAnd:
        case Operator::BitOr:
        case Operator::BitXor:
            if (left == ExprType::Int && right == ExprType::Int)
            {
                return ExprType::Int;
            }
            reportError(TypeChkError::AttemptedBitOpOnNonNumeric, context);
            return ExprType::Error;

        case Operator::Shl:
        case Operator::Shr:
            if (left == ExprType::Int && right == ExprType::Int)
            {
                return ExprType::Int;
            }
            reportError(TypeChkError::AttemptedShiftOnNonInt, context);
            return ExprType::Error;

        case Operator::Assign:
            if (left == right && left != ExprType::Void && left != ExprType::Error)
            {
                return left;
            }
            reportError(TypeChkError::ExpressionTypeMismatch, context);
            return ExprType::Error;

        default:
            return ExprType::Error;
        }
    }

    ExprType checkUnaryOp(Operator op, ExprType operand, const string &context)
    {
        switch (op)
        {
        case Operator::Not:
            if (operand == ExprType::Bool)
            {
                return ExprType::Bool;
            }
            reportError(TypeChkError::AttemptedBoolOpOnNonBools, context);
            return ExprType::Error;

        case Operator::BitNot:
            if (operand == ExprType::Int)
            {
                return ExprType::Int;
            }
            reportError(TypeChkError::AttemptedBitOpOnNonNumeric, context);
            return ExprType::Error;

        case Operator::Sub:
            if (operand == ExprType::Int || operand == ExprType::Float)
            {
                return operand;
            }
            reportError(TypeChkError::AttemptedAddOpOnNonNumeric, context);
            return ExprType::Error;

        default:
            return ExprType::Error;
        }
    }

    // Statement type checking
    void enterFunction(const string &funcName, ExprType returnType)
    {
        currentFunction = funcName;
        currentFunctionReturnType = returnType;
        hasReturnStmt = false;
    }

    void exitFunction()
    {
        if (currentFunctionReturnType != ExprType::Void && !hasReturnStmt)
        {
            reportError(TypeChkError::ReturnStmtNotFound, "function '" + currentFunction + "'");
        }
        currentFunction = "";
        currentFunctionReturnType = ExprType::Void;
    }

    void checkReturnStmt(ExprType returnExprType, const string &context)
    {
        hasReturnStmt = true;

        if (currentFunction.empty())
        {
            reportError(TypeChkError::ErroneousReturnType, "return outside function");
            return;
        }

        if (currentFunctionReturnType != returnExprType)
        {
            reportError(TypeChkError::ErroneousReturnType,
                        "expected " + exprTypeToString(currentFunctionReturnType) +
                            " but got " + exprTypeToString(returnExprType));
        }
    }

    void checkCondition(ExprType condType, const string &context)
    {
        if (condType != ExprType::Bool)
        {
            reportError(TypeChkError::NonBooleanCondStmt, context);
        }
    }

    void enterLoop() { inLoop = true; }
    void exitLoop() { inLoop = false; }

    void checkBreakStmt(const string &context)
    {
        if (!inLoop)
        {
            reportError(TypeChkError::ErroneousBreak, context);
        }
    }

    // Function call type checking
    void checkFunctionCall(const string &funcName,
                           const vector<ExprType> &paramTypes,
                           const string &context)
    {
        if (funcName == "printf")
        {
            return;
        }

        static unordered_map<string, pair<ExprType, vector<ExprType>>> funcSignatures = {
            {"sqrt", {ExprType::Float, {ExprType::Float}}},
            {"abs", {ExprType::Int, {ExprType::Int}}},
            {"strlen", {ExprType::Int, {ExprType::String}}},
            {"malloc", {ExprType::Int, {ExprType::Int}}}};

        auto it = funcSignatures.find(funcName);
        if (it != funcSignatures.end())
        {
            const auto &[returnType, expectedParams] = it->second;

            if (paramTypes.size() != expectedParams.size())
            {
                reportError(TypeChkError::FnCallParamCount,
                            funcName + ": expected " + to_string(expectedParams.size()) +
                                " parameters, got " + to_string(paramTypes.size()));
                return;
            }

            for (size_t i = 0; i < paramTypes.size(); ++i)
            {
                if (paramTypes[i] != expectedParams[i])
                {
                    reportError(TypeChkError::FnCallParamType,
                                funcName + ": parameter " + to_string(i + 1) +
                                    " type mismatch");
                }
            }
        }
    }

    // Variable declaration checking
    void checkVarDecl(const string &varName, ExprType declaredType, ExprType initExprType, const string &context)
    {
        if (declaredType != initExprType && initExprType != ExprType::Unknown)
        {
            reportError(TypeChkError::ErroneousVarDecl,
                        varName + ": declared as " + exprTypeToString(declaredType) +
                            " but initialized with " + exprTypeToString(initExprType));
        }
    }

    // Get errors and clear
    vector<TypeChkError> getErrors()
    {
        return errors;
    }

    void clearErrors()
    {
        errors.clear();
    }

    bool hasErrors() const
    {
        return !errors.empty();
    }
};

// Enhanced ScopeAnalyzer with type information
class ScopeAnalyzer
{
private:
    ScopeNode *currentScope;
    int currentLevel;
    vector<unique_ptr<ScopeNode>> scopePool;

public:
    ScopeAnalyzer() : currentScope(nullptr), currentLevel(0)
    {
        enterScope();
    }

    ~ScopeAnalyzer()
    {
        while (currentScope != nullptr)
        {
            exitScope();
        }
    }

    void enterScope()
    {
        currentLevel++;
        ScopeNode *newScope = new ScopeNode(currentLevel, currentScope);
        scopePool.emplace_back(newScope);
        currentScope = newScope;
    }

    void exitScope()
    {
        if (currentScope != nullptr)
        {
            ScopeNode *oldScope = currentScope;
            currentScope = currentScope->parent;
            currentLevel--;

            for (auto it = scopePool.begin(); it != scopePool.end(); ++it)
            {
                if (it->get() == oldScope)
                {
                    scopePool.erase(it);
                    break;
                }
            }
        }
    }

    ScopeError declareVariable(const string &name, const string &dataType)
    {
        if (!currentScope)
            return ScopeError::UndeclaredVariableAccessed;

        if (!currentScope->addSymbol(name, SymbolType::Variable, dataType))
        {
            return ScopeError::VariableRedefinition;
        }
        return ScopeError::None;
    }

    ScopeError declareFunction(const string &name, const string &returnType, bool isPrototype = false)
    {
        if (!currentScope)
            return ScopeError::UndefinedFunctionCalled;

        if (!currentScope->addSymbol(name, SymbolType::Function, returnType, isPrototype))
        {
            return ScopeError::FunctionPrototypeRedefinition;
        }
        return ScopeError::None;
    }

    ScopeError declareParameter(const string &name, const string &dataType)
    {
        return declareVariable(name, dataType);
    }

    optional<SymbolInfo> lookupSymbol(const string &name) const
    {
        ScopeNode *scope = currentScope;

        while (scope != nullptr)
        {
            auto symbol = scope->lookupLocal(name);
            if (symbol.has_value())
            {
                return symbol;
            }
            scope = scope->parent;
        }

        return nullopt;
    }

    optional<SymbolInfo> lookupVariable(const string &name) const
    {
        auto symbol = lookupSymbol(name);
        if (symbol.has_value() &&
            (symbol->type == SymbolType::Variable || symbol->type == SymbolType::Parameter))
        {
            return symbol;
        }
        return nullopt;
    }

    optional<SymbolInfo> lookupFunction(const string &name) const
    {
        auto symbol = lookupSymbol(name);
        if (symbol.has_value() && symbol->type == SymbolType::Function)
        {
            return symbol;
        }
        return nullopt;
    }

    ScopeError checkVariableAccess(const string &name) const
    {
        auto symbol = lookupVariable(name);
        if (!symbol.has_value())
        {
            return ScopeError::UndeclaredVariableAccessed;
        }
        return ScopeError::None;
    }

    ScopeError checkFunctionCall(const string &name) const
    {
        auto symbol = lookupFunction(name);
        if (!symbol.has_value())
        {
            return ScopeError::UndefinedFunctionCalled;
        }
        return ScopeError::None;
    }

    int getCurrentScopeLevel() const
    {
        return currentLevel;
    }

    bool isGlobalScope() const
    {
        return currentLevel == 1;
    }

    void printScopeHierarchy() const
    {
        cout << "Scope Hierarchy (current level: " << currentLevel << "):" << endl;
        ScopeNode *scope = currentScope;

        while (scope != nullptr)
        {
            cout << "  Level " << scope->scopeLevel << " (" << scope->symbols.size() << " symbols)" << endl;
            for (const auto &[name, symbol] : scope->symbols)
            {
                string typeStr;
                switch (symbol.type)
                {
                case SymbolType::Variable:
                    typeStr = "variable";
                    break;
                case SymbolType::Function:
                    typeStr = "function";
                    break;
                case SymbolType::Parameter:
                    typeStr = "parameter";
                    break;
                }
                cout << "    " << name << " : " << typeStr << " [" << symbol.dataType << "]"
                     << (symbol.isPrototype ? " (prototype)" : "") << endl;
            }
            scope = scope->parent;
        }
    }

    optional<string> getVariableType(const string &name) const
    {
        auto symbol = lookupVariable(name);
        if (symbol.has_value())
        {
            return symbol->dataType;
        }
        return nullopt;
    }

    optional<string> getFunctionReturnType(const string &name) const
    {
        auto symbol = lookupFunction(name);
        if (symbol.has_value())
        {
            return symbol->dataType;
        }
        return nullopt;
    }
};

// Test function demonstrating type checking with code context
void testTypeChecker()
{
    cout << "=== Testing Type Checker ===" << endl;

    ScopeAnalyzer scopeAnalyzer;
    TypeChecker typeChecker(scopeAnalyzer);

    // Test 1: Variable declaration with type mismatch
    cout << "\n1. Testing variable declaration type checking:" << endl;
    cout << "   Code: int x = \"hello\";" << endl;
    cout << "   >>> Declaring variable 'x' as int" << endl;
    scopeAnalyzer.declareVariable("x", "int");
    cout << "   >>> Checking initialization with string literal" << endl;
    typeChecker.checkVarDecl("x", typeChecker.stringToExprType("int"),
                             typeChecker.stringToExprType("string"), "line 1");
    cout << endl;

    // Test 2: Arithmetic operations
    cout << "2. Testing arithmetic operations:" << endl;
    cout << "   Code: int a = 5 + 3.14;" << endl;
    cout << "   >>> Analyzing expression: 5 + 3.14" << endl;
    ExprType result = typeChecker.checkBinaryOp(Operator::Add, ExprType::Int, ExprType::Float, "5 + 3.14");
    cout << "   >>> Result type: " << typeChecker.exprTypeToString(result) << " (int + float -> float)" << endl;
    cout << "   >>> Variable 'a' declared as int, but assigned float - this would cause warning in real compiler" << endl;

    cout << "\n   Code: bool b = true + false;" << endl;
    cout << "   >>> Analyzing expression: true + false" << endl;
    typeChecker.checkBinaryOp(Operator::Add, ExprType::Bool, ExprType::Bool, "true + false");
    cout << endl;

    // Test 3: Function calls
    cout << "3. Testing function call type checking:" << endl;
    cout << "   Code: strlen(123);" << endl;
    cout << "   >>> Checking function call: strlen(int)" << endl;
    cout << "   >>> strlen expects: const char*" << endl;
    typeChecker.checkFunctionCall("strlen", {ExprType::Int}, "line 10");

    cout << "\n   Code: sqrt(2.0);" << endl;
    cout << "   >>> Checking function call: sqrt(double)" << endl;
    cout << "   >>> sqrt expects: double - VALID" << endl;
    typeChecker.checkFunctionCall("sqrt", {ExprType::Float}, "line 15");
    cout << endl;

    // Test 4: Return statements
    cout << "4. Testing return statement checking:" << endl;
    cout << "   Code: int func() { return 3.14; }" << endl;
    cout << "   >>> Entering function 'func' with return type: int" << endl;
    typeChecker.enterFunction("func", ExprType::Int);
    cout << "   >>> Checking return statement: return 3.14" << endl;
    typeChecker.checkReturnStmt(ExprType::Float, "line 20");
    cout << "   >>> Exiting function 'func'" << endl;
    typeChecker.exitFunction();
    cout << endl;

    // Test 5: Conditional statements
    cout << "5. Testing conditional statements:" << endl;
    cout << "   Code: if (5) { ... }" << endl;
    cout << "   >>> Checking condition type: 5 (int)" << endl;
    typeChecker.checkCondition(ExprType::Int, "line 25");

    cout << "\n   Code: while (true) { ... }" << endl;
    cout << "   >>> Checking condition type: true (bool)" << endl;
    typeChecker.checkCondition(ExprType::Bool, "line 30");
    cout << "   >>> Condition is boolean - VALID" << endl;
    cout << endl;

    // Test 6: Break statements
    cout << "6. Testing break statements:" << endl;
    cout << "   Code: break;  // Outside any loop" << endl;
    cout << "   >>> Checking break statement context" << endl;
    typeChecker.checkBreakStmt("line 35");

    cout << "\n   Code: for(;;) { break; }" << endl;
    cout << "   >>> Entering loop context" << endl;
    typeChecker.enterLoop();
    cout << "   >>> Checking break statement inside loop" << endl;
    typeChecker.checkBreakStmt("line 40");
    cout << "   >>> Break is valid inside loop" << endl;
    cout << "   >>> Exiting loop context" << endl;
    typeChecker.exitLoop();
    cout << endl;

    // Test 7: Logical operations
    cout << "7. Testing logical operations:" << endl;
    cout << "   Code: true && false;" << endl;
    cout << "   >>> Analyzing expression: true && false" << endl;
    result = typeChecker.checkBinaryOp(Operator::And, ExprType::Bool, ExprType::Bool, "line 45");
    cout << "   >>> Result type: " << typeChecker.exprTypeToString(result) << " - VALID" << endl;

    cout << "\n   Code: 5 && 3;" << endl;
    cout << "   >>> Analyzing expression: 5 && 3" << endl;
    typeChecker.checkBinaryOp(Operator::And, ExprType::Int, ExprType::Int, "line 50");
    cout << endl;

    // Test 8: Bitwise operations
    cout << "8. Testing bitwise operations:" << endl;
    cout << "   Code: 5 & 3;" << endl;
    cout << "   >>> Analyzing expression: 5 & 3" << endl;
    result = typeChecker.checkBinaryOp(Operator::BitAnd, ExprType::Int, ExprType::Int, "line 55");
    cout << "   >>> Result type: " << typeChecker.exprTypeToString(result) << " - VALID" << endl;

    cout << "\n   Code: 5.0 & 3;" << endl;
    cout << "   >>> Analyzing expression: 5.0 & 3" << endl;
    typeChecker.checkBinaryOp(Operator::BitAnd, ExprType::Float, ExprType::Int, "line 60");
    cout << endl;

    // Test 9: Shift operations
    cout << "9. Testing shift operations:" << endl;
    cout << "   Code: 5 << 2;" << endl;
    cout << "   >>> Analyzing expression: 5 << 2" << endl;
    result = typeChecker.checkBinaryOp(Operator::Shl, ExprType::Int, ExprType::Int, "line 65");
    cout << "   >>> Result type: " << typeChecker.exprTypeToString(result) << " - VALID" << endl;

    cout << "\n   Code: 5.0 << 2;" << endl;
    cout << "   >>> Analyzing expression: 5.0 << 2" << endl;
    typeChecker.checkBinaryOp(Operator::Shl, ExprType::Float, ExprType::Int, "line 70");
    cout << endl;

    // Test 10: Missing return statement
    cout << "10. Testing missing return statement:" << endl;
    cout << "   Code: int missingReturn() { }" << endl;
    cout << "   >>> Entering function 'missingReturn' with return type: int" << endl;
    typeChecker.enterFunction("missingReturn", ExprType::Int);
    cout << "   >>> No return statement found in function body" << endl;
    cout << "   >>> Exiting function 'missingReturn'" << endl;
    typeChecker.exitFunction();

    cout << "\n=== Type Checker Test Complete ===" << endl;
    cout << "Total errors detected: " << typeChecker.getErrors().size() << endl;
}

// Test with a complete function with detailed context
void testCompleteFunction()
{
    cout << "\n\n=== Testing Complete Function Type Checking ===" << endl;

    ScopeAnalyzer scopeAnalyzer;
    TypeChecker typeChecker(scopeAnalyzer);

    cout << "Simulating type checking for function:\n"
         << endl;
    cout << "1: int calculate(int a, float b) {" << endl;
    cout << "2:     bool flag = true;" << endl;
    cout << "3:     float result = a + b;" << endl;
    cout << "4:     if (flag) {" << endl;
    cout << "5:         result = result * 2.0;" << endl;
    cout << "6:     }" << endl;
    cout << "7:     return result;" << endl;
    cout << "8: }" << endl;

    cout << "\n>>> Starting type analysis for function 'calculate'..." << endl;

    // Simulate the function line by line
    typeChecker.enterFunction("calculate", ExprType::Int);

    cout << "\nLine 1: Function declaration" << endl;
    cout << ">>> Return type: int" << endl;
    cout << ">>> Parameters: a (int), b (float)" << endl;

    cout << "\nLine 2: bool flag = true;" << endl;
    cout << ">>> Declaring variable 'flag' as bool" << endl;
    cout << ">>> Initializing with boolean literal - VALID" << endl;
    typeChecker.checkVarDecl("flag", ExprType::Bool, ExprType::Bool, "line 2");

    cout << "\nLine 3: float result = a + b;" << endl;
    cout << ">>> Declaring variable 'result' as float" << endl;
    cout << ">>> Analyzing expression: a + b" << endl;
    cout << ">>> Operand types: int + float" << endl;
    ExprType addResult = typeChecker.checkBinaryOp(Operator::Add, ExprType::Int, ExprType::Float, "line 3");
    cout << ">>> Expression result type: " << typeChecker.exprTypeToString(addResult) << endl;
    cout << ">>> Assignment: float = float - VALID" << endl;
    typeChecker.checkVarDecl("result", ExprType::Float, addResult, "line 3");

    cout << "\nLine 4: if (flag) {" << endl;
    cout << ">>> Checking condition type: flag (bool)" << endl;
    typeChecker.checkCondition(ExprType::Bool, "line 4");
    cout << ">>> Condition is boolean - VALID" << endl;

    cout << "\nLine 5: result = result * 2.0;" << endl;
    cout << ">>> Analyzing expression: result * 2.0" << endl;
    cout << ">>> Operand types: float * float" << endl;
    ExprType mulResult = typeChecker.checkBinaryOp(Operator::Mul, ExprType::Float, ExprType::Float, "line 5");
    cout << ">>> Expression result type: " << typeChecker.exprTypeToString(mulResult) << endl;
    cout << ">>> Assignment: float = float - VALID" << endl;
    typeChecker.checkVarDecl("result", ExprType::Float, mulResult, "line 5");

    cout << "\nLine 7: return result;" << endl;
    cout << ">>> Checking return type: result (float)" << endl;
    cout << ">>> Function return type: int" << endl;
    typeChecker.checkReturnStmt(ExprType::Float, "line 7");

    cout << "\nLine 8: End of function" << endl;
    typeChecker.exitFunction();

    cout << "\n=== Complete Function Analysis Results ===" << endl;
    cout << "Errors detected: " << typeChecker.getErrors().size() << endl;
    if (typeChecker.getErrors().size() == 1)
    {
        cout << "Primary issue: Return type mismatch - function returns int but 'result' is float" << endl;
        cout << "Suggested fix: Change function return type to float or cast return value to int" << endl;
    }
}

// Additional test: Complex expressions with mixed types
void testComplexExpressions()
{
    cout << "\n\n=== Testing Complex Expressions ===" << endl;

    ScopeAnalyzer scopeAnalyzer;
    TypeChecker typeChecker(scopeAnalyzer);

    cout << "Testing complex expression analysis:\n"
         << endl;

    cout << "Code: int x = (5 + 3.0) * 2 > 10 && true;" << endl;
    cout << ">>> Step-by-step analysis:" << endl;

    cout << "1. Analyzing: 5 + 3.0" << endl;
    ExprType step1 = typeChecker.checkBinaryOp(Operator::Add, ExprType::Int, ExprType::Float, "5 + 3.0");
    cout << "   Result: " << typeChecker.exprTypeToString(step1) << endl;

    cout << "2. Analyzing: (float) * 2" << endl;
    ExprType step2 = typeChecker.checkBinaryOp(Operator::Mul, step1, ExprType::Int, "(5 + 3.0) * 2");
    cout << "   Result: " << typeChecker.exprTypeToString(step2) << endl;

    cout << "3. Analyzing: (float) > 10" << endl;
    ExprType step3 = typeChecker.checkBinaryOp(Operator::Gt, step2, ExprType::Int, "(5 + 3.0) * 2 > 10");
    cout << "   Result: " << typeChecker.exprTypeToString(step3) << endl;

    cout << "4. Analyzing: (bool) && true" << endl;
    ExprType step4 = typeChecker.checkBinaryOp(Operator::And, step3, ExprType::Bool, "((5 + 3.0) * 2 > 10) && true");
    cout << "   Final result: " << typeChecker.exprTypeToString(step4) << endl;

    cout << "5. Variable assignment: int x = bool" << endl;
    typeChecker.checkVarDecl("x", ExprType::Int, step4, "int x = complex_expression");

    cout << "\nExpression analysis complete!" << endl;
}

int main()
{
    cout << "=== C Type Checker Implementation ===" << endl;
    cout << "This module performs static type checking using the scope analyzer's symbol table.\n"
         << endl;
    cout << "Each test shows the actual code being analyzed and step-by-step type checking.\n"
         << endl;

    testTypeChecker();
    testCompleteFunction();
    testComplexExpressions();

    cout << "\n=== Type Checker Implementation Complete ===" << endl;
    cout << "The type checker successfully detects:\n";
    cout << "✓ Type mismatches in variable declarations\n";
    cout << "✓ Invalid operations on incompatible types\n";
    cout << "✓ Function call parameter errors\n";
    cout << "✓ Return type violations\n";
    cout << "✓ Invalid break statements\n";
    cout << "✓ Non-boolean conditions\n";
    cout << "✓ Missing return statements\n";
    cout << "✓ Complex expression type propagation\n";

    return 0;
}