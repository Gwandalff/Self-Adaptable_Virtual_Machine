package miniJava.visitor.miniJava;

import miniJava.visitor.miniJava.impl.MiniJavaFactoryImpl;
import org.eclipse.emf.ecore.EFactory;

public interface MiniJavaFactory extends EFactory {
	MiniJavaFactory eINSTANCE = MiniJavaFactoryImpl.init();

	Program createProgram();

	Import createImport();

	TypeDeclaration createTypeDeclaration();

	Clazz createClazz();

	Interface createInterface();

	Member createMember();

	Method createMethod();

	Parameter createParameter();

	Field createField();

	Block createBlock();

	Statement createStatement();

	PrintStatement createPrintStatement();

	LoadImage createLoadImage();

	WriteImage createWriteImage();

	Return createReturn();

	IfStatement createIfStatement();

	WhileStatement createWhileStatement();

	ForStatement createForStatement();

	TypeRef createTypeRef();

	SingleTypeRef createSingleTypeRef();

	ClassRef createClassRef();

	NamedElement createNamedElement();

	TypedDeclaration createTypedDeclaration();

	Symbol createSymbol();

	VariableDeclaration createVariableDeclaration();

	Assignment createAssignment();

	Assignee createAssignee();

	Expression createExpression();

	ArrayTypeRef createArrayTypeRef();

	IntegerTypeRef createIntegerTypeRef();

	BooleanTypeRef createBooleanTypeRef();

	StringTypeRef createStringTypeRef();

	VoidTypeRef createVoidTypeRef();

	Or createOr();

	And createAnd();

	Equality createEquality();

	Inequality createInequality();

	SuperiorOrEqual createSuperiorOrEqual();

	InferiorOrEqual createInferiorOrEqual();

	Superior createSuperior();

	Inferior createInferior();

	Plus createPlus();

	Minus createMinus();

	Multiplication createMultiplication();

	Division createDivision();

	ArrayAccess createArrayAccess();

	ArrayLength createArrayLength();

	Not createNot();

	Neg createNeg();

	FieldAccess createFieldAccess();

	MethodCall createMethodCall();

	StringConstant createStringConstant();

	IntConstant createIntConstant();

	BoolConstant createBoolConstant();

	This createThis();

	Super createSuper();

	Null createNull();

	NewObject createNewObject();

	NewArray createNewArray();

	SymbolRef createSymbolRef();

	Context createContext();

	Value createValue();

	IntegerValue createIntegerValue();

	SymbolBinding createSymbolBinding();

	FieldBinding createFieldBinding();

	StringValue createStringValue();

	BooleanValue createBooleanValue();

	OutputStream createOutputStream();

	State createState();

	Frame createFrame();

	NullValue createNullValue();

	NewCall createNewCall();

	MethodCall2 createMethodCall2();

	ObjectInstance createObjectInstance();

	ArrayInstance createArrayInstance();

	ObjectRefValue createObjectRefValue();

	ArrayRefValue createArrayRefValue();

	Modulo createModulo();

	Sqrt createSqrt();

	MiniJavaPackage getMiniJavaPackage();
}
