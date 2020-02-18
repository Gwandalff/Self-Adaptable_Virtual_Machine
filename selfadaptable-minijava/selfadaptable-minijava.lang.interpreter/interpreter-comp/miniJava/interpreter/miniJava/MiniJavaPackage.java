package miniJava.interpreter.miniJava;

import java.lang.String;
import miniJava.interpreter.miniJava.impl.MiniJavaPackageImpl;
import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EEnum;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

public interface MiniJavaPackage extends EPackage {
	String eNAME = "miniJava";

	String eNS_URI = "http://miniJava.miniJava.miniJava/";

	String eNS_PREFIX = "miniJava";

	MiniJavaPackage eINSTANCE = MiniJavaPackageImpl.init();

	int PROGRAM = 0;

	int PROGRAM__NAME = 0;

	int PROGRAM__IMPORTS = 1;

	int PROGRAM__CLASSES = 2;

	int PROGRAM__STATE = 3;

	int PROGRAM_FEATURE_COUNT = 4;

	int PROGRAM_OPERATION_COUNT = 0;

	int IMPORT = 1;

	int IMPORT__IMPORTED_NAMESPACE = 0;

	int IMPORT_FEATURE_COUNT = 1;

	int IMPORT_OPERATION_COUNT = 0;

	int NAMED_ELEMENT = 2;

	int NAMED_ELEMENT__NAME = 0;

	int NAMED_ELEMENT_FEATURE_COUNT = 1;

	int NAMED_ELEMENT_OPERATION_COUNT = 0;

	int TYPE_DECLARATION = 3;

	int TYPE_DECLARATION__NAME = NAMED_ELEMENT__NAME;

	int TYPE_DECLARATION__ACCESS_LEVEL = 1;

	int TYPE_DECLARATION__IMPLEMENTZ = 2;

	int TYPE_DECLARATION__MEMBERS = 3;

	int TYPE_DECLARATION_FEATURE_COUNT = NAMED_ELEMENT_FEATURE_COUNT + 3;

	int TYPE_DECLARATION_OPERATION_COUNT = NAMED_ELEMENT_OPERATION_COUNT + 0;

	int CLAZZ = 4;

	int CLAZZ__NAME = NAMED_ELEMENT__NAME;

	int CLAZZ__ACCESS_LEVEL = TYPE_DECLARATION__ACCESS_LEVEL;

	int CLAZZ__IMPLEMENTZ = TYPE_DECLARATION__IMPLEMENTZ;

	int CLAZZ__MEMBERS = TYPE_DECLARATION__MEMBERS;

	int CLAZZ__ISABSTRACT = 4;

	int CLAZZ__SUPER_CLASS = 5;

	int CLAZZ_FEATURE_COUNT = TYPE_DECLARATION_FEATURE_COUNT + 2;

	int CLAZZ_OPERATION_COUNT = TYPE_DECLARATION_OPERATION_COUNT + 0;

	int INTERFACE = 5;

	int INTERFACE__NAME = NAMED_ELEMENT__NAME;

	int INTERFACE__ACCESS_LEVEL = TYPE_DECLARATION__ACCESS_LEVEL;

	int INTERFACE__IMPLEMENTZ = TYPE_DECLARATION__IMPLEMENTZ;

	int INTERFACE__MEMBERS = TYPE_DECLARATION__MEMBERS;

	int INTERFACE_FEATURE_COUNT = TYPE_DECLARATION_FEATURE_COUNT + 0;

	int INTERFACE_OPERATION_COUNT = TYPE_DECLARATION_OPERATION_COUNT + 0;

	int TYPED_DECLARATION = 6;

	int TYPED_DECLARATION__NAME = NAMED_ELEMENT__NAME;

	int TYPED_DECLARATION__TYPE_REF = 1;

	int TYPED_DECLARATION_FEATURE_COUNT = NAMED_ELEMENT_FEATURE_COUNT + 1;

	int TYPED_DECLARATION_OPERATION_COUNT = NAMED_ELEMENT_OPERATION_COUNT + 0;

	int MEMBER = 7;

	int MEMBER__NAME = NAMED_ELEMENT__NAME;

	int MEMBER__TYPE_REF = TYPED_DECLARATION__TYPE_REF;

	int MEMBER__ACCESS = 2;

	int MEMBER_FEATURE_COUNT = TYPED_DECLARATION_FEATURE_COUNT + 1;

	int MEMBER_OPERATION_COUNT = TYPED_DECLARATION_OPERATION_COUNT + 0;

	int METHOD = 8;

	int METHOD__NAME = NAMED_ELEMENT__NAME;

	int METHOD__TYPE_REF = TYPED_DECLARATION__TYPE_REF;

	int METHOD__ACCESS = MEMBER__ACCESS;

	int METHOD__ISABSTRACT = 3;

	int METHOD__ISSTATIC = 4;

	int METHOD__PARAMS = 5;

	int METHOD__BODY = 6;

	int METHOD__CACHE = 7;

	int METHOD_FEATURE_COUNT = MEMBER_FEATURE_COUNT + 5;

	int METHOD_OPERATION_COUNT = MEMBER_OPERATION_COUNT + 0;

	int SYMBOL = 9;

	int SYMBOL__NAME = NAMED_ELEMENT__NAME;

	int SYMBOL__TYPE_REF = TYPED_DECLARATION__TYPE_REF;

	int SYMBOL_FEATURE_COUNT = TYPED_DECLARATION_FEATURE_COUNT + 0;

	int SYMBOL_OPERATION_COUNT = TYPED_DECLARATION_OPERATION_COUNT + 0;

	int PARAMETER = 10;

	int PARAMETER__NAME = NAMED_ELEMENT__NAME;

	int PARAMETER__TYPE_REF = TYPED_DECLARATION__TYPE_REF;

	int PARAMETER_FEATURE_COUNT = SYMBOL_FEATURE_COUNT + 0;

	int PARAMETER_OPERATION_COUNT = SYMBOL_OPERATION_COUNT + 0;

	int FIELD = 11;

	int FIELD__NAME = NAMED_ELEMENT__NAME;

	int FIELD__TYPE_REF = TYPED_DECLARATION__TYPE_REF;

	int FIELD__ACCESS = MEMBER__ACCESS;

	int FIELD__DEFAULT_VALUE = 3;

	int FIELD_FEATURE_COUNT = MEMBER_FEATURE_COUNT + 1;

	int FIELD_OPERATION_COUNT = MEMBER_OPERATION_COUNT + 0;

	int STATEMENT = 12;

	int STATEMENT_FEATURE_COUNT = 0;

	int STATEMENT_OPERATION_COUNT = 0;

	int BLOCK = 13;

	int BLOCK__STATEMENTS = 0;

	int BLOCK_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 1;

	int BLOCK_OPERATION_COUNT = STATEMENT_OPERATION_COUNT + 0;

	int PRINT_STATEMENT = 14;

	int PRINT_STATEMENT__EXPRESSION = 0;

	int PRINT_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 1;

	int PRINT_STATEMENT_OPERATION_COUNT = STATEMENT_OPERATION_COUNT + 0;

	int RETURN = 15;

	int RETURN__EXPRESSION = 0;

	int RETURN_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 1;

	int RETURN_OPERATION_COUNT = STATEMENT_OPERATION_COUNT + 0;

	int IF_STATEMENT = 16;

	int IF_STATEMENT__EXPRESSION = 0;

	int IF_STATEMENT__THEN_BLOCK = 1;

	int IF_STATEMENT__ELSE_BLOCK = 2;

	int IF_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 3;

	int IF_STATEMENT_OPERATION_COUNT = STATEMENT_OPERATION_COUNT + 0;

	int WHILE_STATEMENT = 17;

	int WHILE_STATEMENT__CONDITION = 0;

	int WHILE_STATEMENT__BLOCK = 1;

	int WHILE_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 2;

	int WHILE_STATEMENT_OPERATION_COUNT = STATEMENT_OPERATION_COUNT + 0;

	int FOR_STATEMENT = 18;

	int FOR_STATEMENT__DECLARATION = 0;

	int FOR_STATEMENT__CONDITION = 1;

	int FOR_STATEMENT__PROGRESSION = 2;

	int FOR_STATEMENT__BLOCK = 3;

	int FOR_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 4;

	int FOR_STATEMENT_OPERATION_COUNT = STATEMENT_OPERATION_COUNT + 0;

	int TYPE_REF = 19;

	int TYPE_REF_FEATURE_COUNT = 0;

	int TYPE_REF_OPERATION_COUNT = 0;

	int SINGLE_TYPE_REF = 20;

	int SINGLE_TYPE_REF_FEATURE_COUNT = TYPE_REF_FEATURE_COUNT + 0;

	int SINGLE_TYPE_REF_OPERATION_COUNT = TYPE_REF_OPERATION_COUNT + 0;

	int CLASS_REF = 21;

	int CLASS_REF__REFERENCED_CLASS = 0;

	int CLASS_REF_FEATURE_COUNT = SINGLE_TYPE_REF_FEATURE_COUNT + 1;

	int CLASS_REF_OPERATION_COUNT = SINGLE_TYPE_REF_OPERATION_COUNT + 0;

	int VARIABLE_DECLARATION = 22;

	int VARIABLE_DECLARATION__NAME = NAMED_ELEMENT__NAME;

	int VARIABLE_DECLARATION__TYPE_REF = TYPED_DECLARATION__TYPE_REF;

	int VARIABLE_DECLARATION_FEATURE_COUNT = SYMBOL_FEATURE_COUNT + 0;

	int VARIABLE_DECLARATION_OPERATION_COUNT = SYMBOL_OPERATION_COUNT + 0;

	int ASSIGNMENT = 23;

	int ASSIGNMENT__ASSIGNEE = 0;

	int ASSIGNMENT__VALUE = 1;

	int ASSIGNMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 2;

	int ASSIGNMENT_OPERATION_COUNT = STATEMENT_OPERATION_COUNT + 0;

	int ASSIGNEE = 24;

	int ASSIGNEE_FEATURE_COUNT = 0;

	int ASSIGNEE_OPERATION_COUNT = 0;

	int EXPRESSION = 25;

	int EXPRESSION_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 0;

	int EXPRESSION_OPERATION_COUNT = STATEMENT_OPERATION_COUNT + 0;

	int ARRAY_TYPE_REF = 26;

	int ARRAY_TYPE_REF__TYPE_REF = 0;

	int ARRAY_TYPE_REF_FEATURE_COUNT = TYPE_REF_FEATURE_COUNT + 1;

	int ARRAY_TYPE_REF_OPERATION_COUNT = TYPE_REF_OPERATION_COUNT + 0;

	int INTEGER_TYPE_REF = 27;

	int INTEGER_TYPE_REF_FEATURE_COUNT = SINGLE_TYPE_REF_FEATURE_COUNT + 0;

	int INTEGER_TYPE_REF_OPERATION_COUNT = SINGLE_TYPE_REF_OPERATION_COUNT + 0;

	int BOOLEAN_TYPE_REF = 28;

	int BOOLEAN_TYPE_REF_FEATURE_COUNT = SINGLE_TYPE_REF_FEATURE_COUNT + 0;

	int BOOLEAN_TYPE_REF_OPERATION_COUNT = SINGLE_TYPE_REF_OPERATION_COUNT + 0;

	int STRING_TYPE_REF = 29;

	int STRING_TYPE_REF_FEATURE_COUNT = SINGLE_TYPE_REF_FEATURE_COUNT + 0;

	int STRING_TYPE_REF_OPERATION_COUNT = SINGLE_TYPE_REF_OPERATION_COUNT + 0;

	int VOID_TYPE_REF = 30;

	int VOID_TYPE_REF_FEATURE_COUNT = SINGLE_TYPE_REF_FEATURE_COUNT + 0;

	int VOID_TYPE_REF_OPERATION_COUNT = SINGLE_TYPE_REF_OPERATION_COUNT + 0;

	int OR = 31;

	int OR__LEFT = 0;

	int OR__RIGHT = 1;

	int OR_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int OR_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int AND = 32;

	int AND__LEFT = 0;

	int AND__RIGHT = 1;

	int AND_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int AND_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int EQUALITY = 33;

	int EQUALITY__LEFT = 0;

	int EQUALITY__RIGHT = 1;

	int EQUALITY_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int EQUALITY_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int INEQUALITY = 34;

	int INEQUALITY__LEFT = 0;

	int INEQUALITY__RIGHT = 1;

	int INEQUALITY_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int INEQUALITY_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int SUPERIOR_OR_EQUAL = 35;

	int SUPERIOR_OR_EQUAL__LEFT = 0;

	int SUPERIOR_OR_EQUAL__RIGHT = 1;

	int SUPERIOR_OR_EQUAL_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int SUPERIOR_OR_EQUAL_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int INFERIOR_OR_EQUAL = 36;

	int INFERIOR_OR_EQUAL__LEFT = 0;

	int INFERIOR_OR_EQUAL__RIGHT = 1;

	int INFERIOR_OR_EQUAL_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int INFERIOR_OR_EQUAL_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int SUPERIOR = 37;

	int SUPERIOR__LEFT = 0;

	int SUPERIOR__RIGHT = 1;

	int SUPERIOR_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int SUPERIOR_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int INFERIOR = 38;

	int INFERIOR__LEFT = 0;

	int INFERIOR__RIGHT = 1;

	int INFERIOR_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int INFERIOR_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int PLUS = 39;

	int PLUS__LEFT = 0;

	int PLUS__RIGHT = 1;

	int PLUS_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int PLUS_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int MINUS = 40;

	int MINUS__LEFT = 0;

	int MINUS__RIGHT = 1;

	int MINUS_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int MINUS_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int MULTIPLICATION = 41;

	int MULTIPLICATION__LEFT = 0;

	int MULTIPLICATION__RIGHT = 1;

	int MULTIPLICATION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int MULTIPLICATION_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int DIVISION = 42;

	int DIVISION__LEFT = 0;

	int DIVISION__RIGHT = 1;

	int DIVISION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int DIVISION_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int ARRAY_ACCESS = 43;

	int ARRAY_ACCESS__OBJECT = 0;

	int ARRAY_ACCESS__INDEX = 1;

	int ARRAY_ACCESS_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int ARRAY_ACCESS_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int ARRAY_LENGTH = 44;

	int ARRAY_LENGTH__ARRAY = 0;

	int ARRAY_LENGTH_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

	int ARRAY_LENGTH_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int NOT = 45;

	int NOT__EXPRESSION = 0;

	int NOT_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

	int NOT_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int NEG = 46;

	int NEG__EXPRESSION = 0;

	int NEG_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

	int NEG_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int FIELD_ACCESS = 47;

	int FIELD_ACCESS__RECEIVER = 0;

	int FIELD_ACCESS__FIELD = 1;

	int FIELD_ACCESS_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int FIELD_ACCESS_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int METHOD_CALL = 48;

	int METHOD_CALL__RECEIVER = 0;

	int METHOD_CALL__METHOD = 1;

	int METHOD_CALL__ARGS = 2;

	int METHOD_CALL_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 3;

	int METHOD_CALL_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int STRING_CONSTANT = 49;

	int STRING_CONSTANT__VALUE = 0;

	int STRING_CONSTANT_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

	int STRING_CONSTANT_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int INT_CONSTANT = 50;

	int INT_CONSTANT__VALUE = 0;

	int INT_CONSTANT_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

	int INT_CONSTANT_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int BOOL_CONSTANT = 51;

	int BOOL_CONSTANT__VALUE = 0;

	int BOOL_CONSTANT_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

	int BOOL_CONSTANT_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int THIS = 52;

	int THIS_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 0;

	int THIS_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int SUPER = 53;

	int SUPER_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 0;

	int SUPER_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int NULL = 54;

	int NULL_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 0;

	int NULL_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int NEW_OBJECT = 55;

	int NEW_OBJECT__TYPE = 0;

	int NEW_OBJECT__ARGS = 1;

	int NEW_OBJECT_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int NEW_OBJECT_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int NEW_ARRAY = 56;

	int NEW_ARRAY__TYPE = 0;

	int NEW_ARRAY__SIZE = 1;

	int NEW_ARRAY_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int NEW_ARRAY_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int SYMBOL_REF = 57;

	int SYMBOL_REF__SYMBOL = 0;

	int SYMBOL_REF_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

	int SYMBOL_REF_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int CONTEXT = 58;

	int CONTEXT__BINDINGS = 0;

	int CONTEXT__PARENT_CONTEXT = 1;

	int CONTEXT__CHILD_CONTEXT = 2;

	int CONTEXT__CACHE = 3;

	int CONTEXT_FEATURE_COUNT = 4;

	int CONTEXT_OPERATION_COUNT = 0;

	int VALUE = 59;

	int VALUE_FEATURE_COUNT = 0;

	int VALUE_OPERATION_COUNT = 0;

	int INTEGER_VALUE = 60;

	int INTEGER_VALUE__VALUE = 0;

	int INTEGER_VALUE_FEATURE_COUNT = VALUE_FEATURE_COUNT + 1;

	int INTEGER_VALUE_OPERATION_COUNT = VALUE_OPERATION_COUNT + 0;

	int SYMBOL_BINDING = 61;

	int SYMBOL_BINDING__VALUE = 0;

	int SYMBOL_BINDING__SYMBOL = 1;

	int SYMBOL_BINDING_FEATURE_COUNT = 2;

	int SYMBOL_BINDING_OPERATION_COUNT = 0;

	int FIELD_BINDING = 62;

	int FIELD_BINDING__FIELD = 0;

	int FIELD_BINDING__VALUE = 1;

	int FIELD_BINDING_FEATURE_COUNT = 2;

	int FIELD_BINDING_OPERATION_COUNT = 0;

	int STRING_VALUE = 63;

	int STRING_VALUE__VALUE = 0;

	int STRING_VALUE_FEATURE_COUNT = VALUE_FEATURE_COUNT + 1;

	int STRING_VALUE_OPERATION_COUNT = VALUE_OPERATION_COUNT + 0;

	int BOOLEAN_VALUE = 64;

	int BOOLEAN_VALUE__VALUE = 0;

	int BOOLEAN_VALUE_FEATURE_COUNT = VALUE_FEATURE_COUNT + 1;

	int BOOLEAN_VALUE_OPERATION_COUNT = VALUE_OPERATION_COUNT + 0;

	int OUTPUT_STREAM = 65;

	int OUTPUT_STREAM__STREAM = 0;

	int OUTPUT_STREAM_FEATURE_COUNT = 1;

	int OUTPUT_STREAM_OPERATION_COUNT = 0;

	int STATE = 66;

	int STATE__ROOT_FRAME = 0;

	int STATE__OBJECTS_HEAP = 1;

	int STATE__OUTPUT_STREAM = 2;

	int STATE__ARRAYS_HEAP = 3;

	int STATE__CONTEXT_CACHE = 4;

	int STATE__FRAME_CACHE = 5;

	int STATE_FEATURE_COUNT = 6;

	int STATE_OPERATION_COUNT = 0;

	int FRAME = 67;

	int FRAME__CALL = 0;

	int FRAME__INSTANCE = 1;

	int FRAME__CHILD_FRAME = 2;

	int FRAME__PARENT_FRAME = 3;

	int FRAME__ROOT_CONTEXT = 4;

	int FRAME__RETURN_VALUE = 5;

	int FRAME_FEATURE_COUNT = 6;

	int FRAME_OPERATION_COUNT = 0;

	int NULL_VALUE = 68;

	int NULL_VALUE_FEATURE_COUNT = VALUE_FEATURE_COUNT + 0;

	int NULL_VALUE_OPERATION_COUNT = VALUE_OPERATION_COUNT + 0;

	int CALL = 69;

	int CALL_FEATURE_COUNT = 0;

	int CALL_OPERATION_COUNT = 0;

	int NEW_CALL = 70;

	int NEW_CALL__NEWZ = 0;

	int NEW_CALL_FEATURE_COUNT = CALL_FEATURE_COUNT + 1;

	int NEW_CALL_OPERATION_COUNT = CALL_OPERATION_COUNT + 0;

	int METHOD_CALL2 = 71;

	int METHOD_CALL2__METHODCALL = 0;

	int METHOD_CALL2_FEATURE_COUNT = CALL_FEATURE_COUNT + 1;

	int METHOD_CALL2_OPERATION_COUNT = CALL_OPERATION_COUNT + 0;

	int OBJECT_INSTANCE = 72;

	int OBJECT_INSTANCE__FIELDBINDINGS = 0;

	int OBJECT_INSTANCE__TYPE = 1;

	int OBJECT_INSTANCE_FEATURE_COUNT = 2;

	int OBJECT_INSTANCE_OPERATION_COUNT = 0;

	int ARRAY_INSTANCE = 73;

	int ARRAY_INSTANCE__VALUE = 0;

	int ARRAY_INSTANCE__SIZE = 1;

	int ARRAY_INSTANCE_FEATURE_COUNT = 2;

	int ARRAY_INSTANCE_OPERATION_COUNT = 0;

	int OBJECT_REF_VALUE = 74;

	int OBJECT_REF_VALUE__INSTANCE = 0;

	int OBJECT_REF_VALUE_FEATURE_COUNT = VALUE_FEATURE_COUNT + 1;

	int OBJECT_REF_VALUE_OPERATION_COUNT = VALUE_OPERATION_COUNT + 0;

	int ARRAY_REF_VALUE = 75;

	int ARRAY_REF_VALUE__INSTANCE = 0;

	int ARRAY_REF_VALUE_FEATURE_COUNT = VALUE_FEATURE_COUNT + 1;

	int ARRAY_REF_VALUE_OPERATION_COUNT = VALUE_OPERATION_COUNT + 0;

	int SYMBOL_TO_SYMBOL_BINDING_MAP = 76;

	int SYMBOL_TO_SYMBOL_BINDING_MAP__KEY = 0;

	int SYMBOL_TO_SYMBOL_BINDING_MAP__VALUE = 1;

	int SYMBOL_TO_SYMBOL_BINDING_MAP_FEATURE_COUNT = 2;

	int SYMBOL_TO_SYMBOL_BINDING_MAP_OPERATION_COUNT = 0;

	int CLAZZ_TO_METHOD_MAP = 77;

	int CLAZZ_TO_METHOD_MAP__KEY = 0;

	int CLAZZ_TO_METHOD_MAP__VALUE = 1;

	int CLAZZ_TO_METHOD_MAP_FEATURE_COUNT = 2;

	int CLAZZ_TO_METHOD_MAP_OPERATION_COUNT = 0;

	int MODULO = 78;

	int MODULO__LEFT = 0;

	int MODULO__RIGHT = 1;

	int MODULO_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 2;

	int MODULO_OPERATION_COUNT = EXPRESSION_OPERATION_COUNT + 0;

	int ACCESS_LEVEL = 79;

	EClass getProgram();

	EAttribute getProgram_Name();

	EReference getProgram_Imports();

	EReference getProgram_Classes();

	EReference getProgram_State();

	EClass getImport();

	EAttribute getImport_ImportedNamespace();

	EClass getTypeDeclaration();

	EAttribute getTypeDeclaration_AccessLevel();

	EReference getTypeDeclaration_Implementz();

	EReference getTypeDeclaration_Members();

	EClass getClazz();

	EAttribute getClazz_Isabstract();

	EReference getClazz_SuperClass();

	EClass getInterface();

	EClass getMember();

	EAttribute getMember_Access();

	EClass getMethod();

	EAttribute getMethod_Isabstract();

	EAttribute getMethod_Isstatic();

	EReference getMethod_Params();

	EReference getMethod_Body();

	EReference getMethod_Cache();

	EClass getParameter();

	EClass getField();

	EReference getField_DefaultValue();

	EEnum getAccessLevel();

	EClass getBlock();

	EReference getBlock_Statements();

	EClass getStatement();

	EClass getPrintStatement();

	EReference getPrintStatement_Expression();

	EClass getReturn();

	EReference getReturn_Expression();

	EClass getIfStatement();

	EReference getIfStatement_Expression();

	EReference getIfStatement_ThenBlock();

	EReference getIfStatement_ElseBlock();

	EClass getWhileStatement();

	EReference getWhileStatement_Condition();

	EReference getWhileStatement_Block();

	EClass getForStatement();

	EReference getForStatement_Declaration();

	EReference getForStatement_Condition();

	EReference getForStatement_Progression();

	EReference getForStatement_Block();

	EClass getTypeRef();

	EClass getSingleTypeRef();

	EClass getClassRef();

	EReference getClassRef_ReferencedClass();

	EClass getNamedElement();

	EAttribute getNamedElement_Name();

	EClass getTypedDeclaration();

	EReference getTypedDeclaration_TypeRef();

	EClass getSymbol();

	EClass getVariableDeclaration();

	EClass getAssignment();

	EReference getAssignment_Assignee();

	EReference getAssignment_Value();

	EClass getAssignee();

	EClass getExpression();

	EClass getArrayTypeRef();

	EReference getArrayTypeRef_TypeRef();

	EClass getIntegerTypeRef();

	EClass getBooleanTypeRef();

	EClass getStringTypeRef();

	EClass getVoidTypeRef();

	EClass getOr();

	EReference getOr_Left();

	EReference getOr_Right();

	EClass getAnd();

	EReference getAnd_Left();

	EReference getAnd_Right();

	EClass getEquality();

	EReference getEquality_Left();

	EReference getEquality_Right();

	EClass getInequality();

	EReference getInequality_Left();

	EReference getInequality_Right();

	EClass getSuperiorOrEqual();

	EReference getSuperiorOrEqual_Left();

	EReference getSuperiorOrEqual_Right();

	EClass getInferiorOrEqual();

	EReference getInferiorOrEqual_Left();

	EReference getInferiorOrEqual_Right();

	EClass getSuperior();

	EReference getSuperior_Left();

	EReference getSuperior_Right();

	EClass getInferior();

	EReference getInferior_Left();

	EReference getInferior_Right();

	EClass getPlus();

	EReference getPlus_Left();

	EReference getPlus_Right();

	EClass getMinus();

	EReference getMinus_Left();

	EReference getMinus_Right();

	EClass getMultiplication();

	EReference getMultiplication_Left();

	EReference getMultiplication_Right();

	EClass getDivision();

	EReference getDivision_Left();

	EReference getDivision_Right();

	EClass getArrayAccess();

	EReference getArrayAccess_Object();

	EReference getArrayAccess_Index();

	EClass getArrayLength();

	EReference getArrayLength_Array();

	EClass getNot();

	EReference getNot_Expression();

	EClass getNeg();

	EReference getNeg_Expression();

	EClass getFieldAccess();

	EReference getFieldAccess_Receiver();

	EReference getFieldAccess_Field();

	EClass getMethodCall();

	EReference getMethodCall_Receiver();

	EReference getMethodCall_Method();

	EReference getMethodCall_Args();

	EClass getStringConstant();

	EAttribute getStringConstant_Value();

	EClass getIntConstant();

	EAttribute getIntConstant_Value();

	EClass getBoolConstant();

	EAttribute getBoolConstant_Value();

	EClass getThis();

	EClass getSuper();

	EClass getNull();

	EClass getNewObject();

	EReference getNewObject_Type();

	EReference getNewObject_Args();

	EClass getNewArray();

	EReference getNewArray_Type();

	EReference getNewArray_Size();

	EClass getSymbolRef();

	EReference getSymbolRef_Symbol();

	EClass getContext();

	EReference getContext_Bindings();

	EReference getContext_ParentContext();

	EReference getContext_ChildContext();

	EReference getContext_Cache();

	EClass getValue();

	EClass getIntegerValue();

	EAttribute getIntegerValue_Value();

	EClass getSymbolBinding();

	EReference getSymbolBinding_Value();

	EReference getSymbolBinding_Symbol();

	EClass getFieldBinding();

	EReference getFieldBinding_Field();

	EReference getFieldBinding_Value();

	EClass getStringValue();

	EAttribute getStringValue_Value();

	EClass getBooleanValue();

	EAttribute getBooleanValue_Value();

	EClass getOutputStream();

	EAttribute getOutputStream_Stream();

	EClass getState();

	EReference getState_RootFrame();

	EReference getState_ObjectsHeap();

	EReference getState_OutputStream();

	EReference getState_ArraysHeap();

	EReference getState_ContextCache();

	EReference getState_FrameCache();

	EClass getFrame();

	EReference getFrame_Call();

	EReference getFrame_Instance();

	EReference getFrame_ChildFrame();

	EReference getFrame_ParentFrame();

	EReference getFrame_RootContext();

	EReference getFrame_ReturnValue();

	EClass getNullValue();

	EClass getCall();

	EClass getNewCall();

	EReference getNewCall_Newz();

	EClass getMethodCall2();

	EReference getMethodCall2_Methodcall();

	EClass getObjectInstance();

	EReference getObjectInstance_Fieldbindings();

	EReference getObjectInstance_Type();

	EClass getArrayInstance();

	EReference getArrayInstance_Value();

	EAttribute getArrayInstance_Size();

	EClass getObjectRefValue();

	EReference getObjectRefValue_Instance();

	EClass getArrayRefValue();

	EReference getArrayRefValue_Instance();

	EClass getSymbolToSymbolBindingMap();

	EReference getSymbolToSymbolBindingMap_Key();

	EReference getSymbolToSymbolBindingMap_Value();

	EClass getClazzToMethodMap();

	EReference getClazzToMethodMap_Key();

	EReference getClazzToMethodMap_Value();

	EClass getModulo();

	EReference getModulo_Left();

	EReference getModulo_Right();

	MiniJavaFactory getMiniJavaFactory();

	interface Literals {
		EClass PROGRAM = eINSTANCE.getProgram();

		EAttribute PROGRAM__NAME = eINSTANCE.getProgram_Name();

		EReference PROGRAM__IMPORTS = eINSTANCE.getProgram_Imports();

		EReference PROGRAM__CLASSES = eINSTANCE.getProgram_Classes();

		EReference PROGRAM__STATE = eINSTANCE.getProgram_State();

		EClass IMPORT = eINSTANCE.getImport();

		EAttribute IMPORT__IMPORTED_NAMESPACE = eINSTANCE.getImport_ImportedNamespace();

		EClass NAMED_ELEMENT = eINSTANCE.getNamedElement();

		EAttribute NAMED_ELEMENT__NAME = eINSTANCE.getNamedElement_Name();

		EClass TYPE_DECLARATION = eINSTANCE.getTypeDeclaration();

		EAttribute TYPE_DECLARATION__ACCESS_LEVEL = eINSTANCE.getTypeDeclaration_AccessLevel();

		EReference TYPE_DECLARATION__IMPLEMENTZ = eINSTANCE.getTypeDeclaration_Implementz();

		EReference TYPE_DECLARATION__MEMBERS = eINSTANCE.getTypeDeclaration_Members();

		EClass CLAZZ = eINSTANCE.getClazz();

		EAttribute CLAZZ__ISABSTRACT = eINSTANCE.getClazz_Isabstract();

		EReference CLAZZ__SUPER_CLASS = eINSTANCE.getClazz_SuperClass();

		EClass INTERFACE = eINSTANCE.getInterface();

		EClass TYPED_DECLARATION = eINSTANCE.getTypedDeclaration();

		EReference TYPED_DECLARATION__TYPE_REF = eINSTANCE.getTypedDeclaration_TypeRef();

		EClass MEMBER = eINSTANCE.getMember();

		EAttribute MEMBER__ACCESS = eINSTANCE.getMember_Access();

		EClass METHOD = eINSTANCE.getMethod();

		EAttribute METHOD__ISABSTRACT = eINSTANCE.getMethod_Isabstract();

		EAttribute METHOD__ISSTATIC = eINSTANCE.getMethod_Isstatic();

		EReference METHOD__PARAMS = eINSTANCE.getMethod_Params();

		EReference METHOD__BODY = eINSTANCE.getMethod_Body();

		EReference METHOD__CACHE = eINSTANCE.getMethod_Cache();

		EClass SYMBOL = eINSTANCE.getSymbol();

		EClass PARAMETER = eINSTANCE.getParameter();

		EClass FIELD = eINSTANCE.getField();

		EReference FIELD__DEFAULT_VALUE = eINSTANCE.getField_DefaultValue();

		EClass STATEMENT = eINSTANCE.getStatement();

		EClass BLOCK = eINSTANCE.getBlock();

		EReference BLOCK__STATEMENTS = eINSTANCE.getBlock_Statements();

		EClass PRINT_STATEMENT = eINSTANCE.getPrintStatement();

		EReference PRINT_STATEMENT__EXPRESSION = eINSTANCE.getPrintStatement_Expression();

		EClass RETURN = eINSTANCE.getReturn();

		EReference RETURN__EXPRESSION = eINSTANCE.getReturn_Expression();

		EClass IF_STATEMENT = eINSTANCE.getIfStatement();

		EReference IF_STATEMENT__EXPRESSION = eINSTANCE.getIfStatement_Expression();

		EReference IF_STATEMENT__THEN_BLOCK = eINSTANCE.getIfStatement_ThenBlock();

		EReference IF_STATEMENT__ELSE_BLOCK = eINSTANCE.getIfStatement_ElseBlock();

		EClass WHILE_STATEMENT = eINSTANCE.getWhileStatement();

		EReference WHILE_STATEMENT__CONDITION = eINSTANCE.getWhileStatement_Condition();

		EReference WHILE_STATEMENT__BLOCK = eINSTANCE.getWhileStatement_Block();

		EClass FOR_STATEMENT = eINSTANCE.getForStatement();

		EReference FOR_STATEMENT__DECLARATION = eINSTANCE.getForStatement_Declaration();

		EReference FOR_STATEMENT__CONDITION = eINSTANCE.getForStatement_Condition();

		EReference FOR_STATEMENT__PROGRESSION = eINSTANCE.getForStatement_Progression();

		EReference FOR_STATEMENT__BLOCK = eINSTANCE.getForStatement_Block();

		EClass TYPE_REF = eINSTANCE.getTypeRef();

		EClass SINGLE_TYPE_REF = eINSTANCE.getSingleTypeRef();

		EClass CLASS_REF = eINSTANCE.getClassRef();

		EReference CLASS_REF__REFERENCED_CLASS = eINSTANCE.getClassRef_ReferencedClass();

		EClass VARIABLE_DECLARATION = eINSTANCE.getVariableDeclaration();

		EClass ASSIGNMENT = eINSTANCE.getAssignment();

		EReference ASSIGNMENT__ASSIGNEE = eINSTANCE.getAssignment_Assignee();

		EReference ASSIGNMENT__VALUE = eINSTANCE.getAssignment_Value();

		EClass ASSIGNEE = eINSTANCE.getAssignee();

		EClass EXPRESSION = eINSTANCE.getExpression();

		EClass ARRAY_TYPE_REF = eINSTANCE.getArrayTypeRef();

		EReference ARRAY_TYPE_REF__TYPE_REF = eINSTANCE.getArrayTypeRef_TypeRef();

		EClass INTEGER_TYPE_REF = eINSTANCE.getIntegerTypeRef();

		EClass BOOLEAN_TYPE_REF = eINSTANCE.getBooleanTypeRef();

		EClass STRING_TYPE_REF = eINSTANCE.getStringTypeRef();

		EClass VOID_TYPE_REF = eINSTANCE.getVoidTypeRef();

		EClass OR = eINSTANCE.getOr();

		EReference OR__LEFT = eINSTANCE.getOr_Left();

		EReference OR__RIGHT = eINSTANCE.getOr_Right();

		EClass AND = eINSTANCE.getAnd();

		EReference AND__LEFT = eINSTANCE.getAnd_Left();

		EReference AND__RIGHT = eINSTANCE.getAnd_Right();

		EClass EQUALITY = eINSTANCE.getEquality();

		EReference EQUALITY__LEFT = eINSTANCE.getEquality_Left();

		EReference EQUALITY__RIGHT = eINSTANCE.getEquality_Right();

		EClass INEQUALITY = eINSTANCE.getInequality();

		EReference INEQUALITY__LEFT = eINSTANCE.getInequality_Left();

		EReference INEQUALITY__RIGHT = eINSTANCE.getInequality_Right();

		EClass SUPERIOR_OR_EQUAL = eINSTANCE.getSuperiorOrEqual();

		EReference SUPERIOR_OR_EQUAL__LEFT = eINSTANCE.getSuperiorOrEqual_Left();

		EReference SUPERIOR_OR_EQUAL__RIGHT = eINSTANCE.getSuperiorOrEqual_Right();

		EClass INFERIOR_OR_EQUAL = eINSTANCE.getInferiorOrEqual();

		EReference INFERIOR_OR_EQUAL__LEFT = eINSTANCE.getInferiorOrEqual_Left();

		EReference INFERIOR_OR_EQUAL__RIGHT = eINSTANCE.getInferiorOrEqual_Right();

		EClass SUPERIOR = eINSTANCE.getSuperior();

		EReference SUPERIOR__LEFT = eINSTANCE.getSuperior_Left();

		EReference SUPERIOR__RIGHT = eINSTANCE.getSuperior_Right();

		EClass INFERIOR = eINSTANCE.getInferior();

		EReference INFERIOR__LEFT = eINSTANCE.getInferior_Left();

		EReference INFERIOR__RIGHT = eINSTANCE.getInferior_Right();

		EClass PLUS = eINSTANCE.getPlus();

		EReference PLUS__LEFT = eINSTANCE.getPlus_Left();

		EReference PLUS__RIGHT = eINSTANCE.getPlus_Right();

		EClass MINUS = eINSTANCE.getMinus();

		EReference MINUS__LEFT = eINSTANCE.getMinus_Left();

		EReference MINUS__RIGHT = eINSTANCE.getMinus_Right();

		EClass MULTIPLICATION = eINSTANCE.getMultiplication();

		EReference MULTIPLICATION__LEFT = eINSTANCE.getMultiplication_Left();

		EReference MULTIPLICATION__RIGHT = eINSTANCE.getMultiplication_Right();

		EClass DIVISION = eINSTANCE.getDivision();

		EReference DIVISION__LEFT = eINSTANCE.getDivision_Left();

		EReference DIVISION__RIGHT = eINSTANCE.getDivision_Right();

		EClass ARRAY_ACCESS = eINSTANCE.getArrayAccess();

		EReference ARRAY_ACCESS__OBJECT = eINSTANCE.getArrayAccess_Object();

		EReference ARRAY_ACCESS__INDEX = eINSTANCE.getArrayAccess_Index();

		EClass ARRAY_LENGTH = eINSTANCE.getArrayLength();

		EReference ARRAY_LENGTH__ARRAY = eINSTANCE.getArrayLength_Array();

		EClass NOT = eINSTANCE.getNot();

		EReference NOT__EXPRESSION = eINSTANCE.getNot_Expression();

		EClass NEG = eINSTANCE.getNeg();

		EReference NEG__EXPRESSION = eINSTANCE.getNeg_Expression();

		EClass FIELD_ACCESS = eINSTANCE.getFieldAccess();

		EReference FIELD_ACCESS__RECEIVER = eINSTANCE.getFieldAccess_Receiver();

		EReference FIELD_ACCESS__FIELD = eINSTANCE.getFieldAccess_Field();

		EClass METHOD_CALL = eINSTANCE.getMethodCall();

		EReference METHOD_CALL__RECEIVER = eINSTANCE.getMethodCall_Receiver();

		EReference METHOD_CALL__METHOD = eINSTANCE.getMethodCall_Method();

		EReference METHOD_CALL__ARGS = eINSTANCE.getMethodCall_Args();

		EClass STRING_CONSTANT = eINSTANCE.getStringConstant();

		EAttribute STRING_CONSTANT__VALUE = eINSTANCE.getStringConstant_Value();

		EClass INT_CONSTANT = eINSTANCE.getIntConstant();

		EAttribute INT_CONSTANT__VALUE = eINSTANCE.getIntConstant_Value();

		EClass BOOL_CONSTANT = eINSTANCE.getBoolConstant();

		EAttribute BOOL_CONSTANT__VALUE = eINSTANCE.getBoolConstant_Value();

		EClass THIS = eINSTANCE.getThis();

		EClass SUPER = eINSTANCE.getSuper();

		EClass NULL = eINSTANCE.getNull();

		EClass NEW_OBJECT = eINSTANCE.getNewObject();

		EReference NEW_OBJECT__TYPE = eINSTANCE.getNewObject_Type();

		EReference NEW_OBJECT__ARGS = eINSTANCE.getNewObject_Args();

		EClass NEW_ARRAY = eINSTANCE.getNewArray();

		EReference NEW_ARRAY__TYPE = eINSTANCE.getNewArray_Type();

		EReference NEW_ARRAY__SIZE = eINSTANCE.getNewArray_Size();

		EClass SYMBOL_REF = eINSTANCE.getSymbolRef();

		EReference SYMBOL_REF__SYMBOL = eINSTANCE.getSymbolRef_Symbol();

		EClass CONTEXT = eINSTANCE.getContext();

		EReference CONTEXT__BINDINGS = eINSTANCE.getContext_Bindings();

		EReference CONTEXT__PARENT_CONTEXT = eINSTANCE.getContext_ParentContext();

		EReference CONTEXT__CHILD_CONTEXT = eINSTANCE.getContext_ChildContext();

		EReference CONTEXT__CACHE = eINSTANCE.getContext_Cache();

		EClass VALUE = eINSTANCE.getValue();

		EClass INTEGER_VALUE = eINSTANCE.getIntegerValue();

		EAttribute INTEGER_VALUE__VALUE = eINSTANCE.getIntegerValue_Value();

		EClass SYMBOL_BINDING = eINSTANCE.getSymbolBinding();

		EReference SYMBOL_BINDING__VALUE = eINSTANCE.getSymbolBinding_Value();

		EReference SYMBOL_BINDING__SYMBOL = eINSTANCE.getSymbolBinding_Symbol();

		EClass FIELD_BINDING = eINSTANCE.getFieldBinding();

		EReference FIELD_BINDING__FIELD = eINSTANCE.getFieldBinding_Field();

		EReference FIELD_BINDING__VALUE = eINSTANCE.getFieldBinding_Value();

		EClass STRING_VALUE = eINSTANCE.getStringValue();

		EAttribute STRING_VALUE__VALUE = eINSTANCE.getStringValue_Value();

		EClass BOOLEAN_VALUE = eINSTANCE.getBooleanValue();

		EAttribute BOOLEAN_VALUE__VALUE = eINSTANCE.getBooleanValue_Value();

		EClass OUTPUT_STREAM = eINSTANCE.getOutputStream();

		EAttribute OUTPUT_STREAM__STREAM = eINSTANCE.getOutputStream_Stream();

		EClass STATE = eINSTANCE.getState();

		EReference STATE__ROOT_FRAME = eINSTANCE.getState_RootFrame();

		EReference STATE__OBJECTS_HEAP = eINSTANCE.getState_ObjectsHeap();

		EReference STATE__OUTPUT_STREAM = eINSTANCE.getState_OutputStream();

		EReference STATE__ARRAYS_HEAP = eINSTANCE.getState_ArraysHeap();

		EReference STATE__CONTEXT_CACHE = eINSTANCE.getState_ContextCache();

		EReference STATE__FRAME_CACHE = eINSTANCE.getState_FrameCache();

		EClass FRAME = eINSTANCE.getFrame();

		EReference FRAME__CALL = eINSTANCE.getFrame_Call();

		EReference FRAME__INSTANCE = eINSTANCE.getFrame_Instance();

		EReference FRAME__CHILD_FRAME = eINSTANCE.getFrame_ChildFrame();

		EReference FRAME__PARENT_FRAME = eINSTANCE.getFrame_ParentFrame();

		EReference FRAME__ROOT_CONTEXT = eINSTANCE.getFrame_RootContext();

		EReference FRAME__RETURN_VALUE = eINSTANCE.getFrame_ReturnValue();

		EClass NULL_VALUE = eINSTANCE.getNullValue();

		EClass CALL = eINSTANCE.getCall();

		EClass NEW_CALL = eINSTANCE.getNewCall();

		EReference NEW_CALL__NEWZ = eINSTANCE.getNewCall_Newz();

		EClass METHOD_CALL2 = eINSTANCE.getMethodCall2();

		EReference METHOD_CALL2__METHODCALL = eINSTANCE.getMethodCall2_Methodcall();

		EClass OBJECT_INSTANCE = eINSTANCE.getObjectInstance();

		EReference OBJECT_INSTANCE__FIELDBINDINGS = eINSTANCE.getObjectInstance_Fieldbindings();

		EReference OBJECT_INSTANCE__TYPE = eINSTANCE.getObjectInstance_Type();

		EClass ARRAY_INSTANCE = eINSTANCE.getArrayInstance();

		EReference ARRAY_INSTANCE__VALUE = eINSTANCE.getArrayInstance_Value();

		EAttribute ARRAY_INSTANCE__SIZE = eINSTANCE.getArrayInstance_Size();

		EClass OBJECT_REF_VALUE = eINSTANCE.getObjectRefValue();

		EReference OBJECT_REF_VALUE__INSTANCE = eINSTANCE.getObjectRefValue_Instance();

		EClass ARRAY_REF_VALUE = eINSTANCE.getArrayRefValue();

		EReference ARRAY_REF_VALUE__INSTANCE = eINSTANCE.getArrayRefValue_Instance();

		EClass SYMBOL_TO_SYMBOL_BINDING_MAP = eINSTANCE.getSymbolToSymbolBindingMap();

		EReference SYMBOL_TO_SYMBOL_BINDING_MAP__KEY = eINSTANCE.getSymbolToSymbolBindingMap_Key();

		EReference SYMBOL_TO_SYMBOL_BINDING_MAP__VALUE = eINSTANCE.getSymbolToSymbolBindingMap_Value();

		EClass CLAZZ_TO_METHOD_MAP = eINSTANCE.getClazzToMethodMap();

		EReference CLAZZ_TO_METHOD_MAP__KEY = eINSTANCE.getClazzToMethodMap_Key();

		EReference CLAZZ_TO_METHOD_MAP__VALUE = eINSTANCE.getClazzToMethodMap_Value();

		EClass MODULO = eINSTANCE.getModulo();

		EReference MODULO__LEFT = eINSTANCE.getModulo_Left();

		EReference MODULO__RIGHT = eINSTANCE.getModulo_Right();

		EEnum ACCESSLEVEL = eINSTANCE.getAccessLevel();
	}
}
