package miniJava.visitor.miniJava.impl;

import java.lang.Deprecated;
import java.lang.IllegalArgumentException;
import java.lang.Object;
import java.lang.Override;
import java.lang.String;
import java.util.Map;
import miniJava.visitor.miniJava.AccessLevel;
import miniJava.visitor.miniJava.And;
import miniJava.visitor.miniJava.ArrayAccess;
import miniJava.visitor.miniJava.ArrayInstance;
import miniJava.visitor.miniJava.ArrayLength;
import miniJava.visitor.miniJava.ArrayRefValue;
import miniJava.visitor.miniJava.ArrayTypeRef;
import miniJava.visitor.miniJava.Assignee;
import miniJava.visitor.miniJava.Assignment;
import miniJava.visitor.miniJava.Block;
import miniJava.visitor.miniJava.BoolConstant;
import miniJava.visitor.miniJava.BooleanTypeRef;
import miniJava.visitor.miniJava.BooleanValue;
import miniJava.visitor.miniJava.ClassRef;
import miniJava.visitor.miniJava.Clazz;
import miniJava.visitor.miniJava.Context;
import miniJava.visitor.miniJava.Division;
import miniJava.visitor.miniJava.Equality;
import miniJava.visitor.miniJava.Expression;
import miniJava.visitor.miniJava.Field;
import miniJava.visitor.miniJava.FieldAccess;
import miniJava.visitor.miniJava.FieldBinding;
import miniJava.visitor.miniJava.ForStatement;
import miniJava.visitor.miniJava.Frame;
import miniJava.visitor.miniJava.IfStatement;
import miniJava.visitor.miniJava.Import;
import miniJava.visitor.miniJava.Inequality;
import miniJava.visitor.miniJava.Inferior;
import miniJava.visitor.miniJava.InferiorOrEqual;
import miniJava.visitor.miniJava.IntConstant;
import miniJava.visitor.miniJava.IntegerTypeRef;
import miniJava.visitor.miniJava.IntegerValue;
import miniJava.visitor.miniJava.Interface;
import miniJava.visitor.miniJava.LoadImage;
import miniJava.visitor.miniJava.Member;
import miniJava.visitor.miniJava.Method;
import miniJava.visitor.miniJava.MethodCall;
import miniJava.visitor.miniJava.MethodCall2;
import miniJava.visitor.miniJava.MiniJavaFactory;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.Minus;
import miniJava.visitor.miniJava.Modulo;
import miniJava.visitor.miniJava.Multiplication;
import miniJava.visitor.miniJava.NamedElement;
import miniJava.visitor.miniJava.Neg;
import miniJava.visitor.miniJava.NewArray;
import miniJava.visitor.miniJava.NewCall;
import miniJava.visitor.miniJava.NewObject;
import miniJava.visitor.miniJava.Not;
import miniJava.visitor.miniJava.Null;
import miniJava.visitor.miniJava.NullValue;
import miniJava.visitor.miniJava.ObjectInstance;
import miniJava.visitor.miniJava.ObjectRefValue;
import miniJava.visitor.miniJava.Or;
import miniJava.visitor.miniJava.OutputStream;
import miniJava.visitor.miniJava.Parameter;
import miniJava.visitor.miniJava.Plus;
import miniJava.visitor.miniJava.PrintStatement;
import miniJava.visitor.miniJava.Program;
import miniJava.visitor.miniJava.Return;
import miniJava.visitor.miniJava.SingleTypeRef;
import miniJava.visitor.miniJava.Sqrt;
import miniJava.visitor.miniJava.State;
import miniJava.visitor.miniJava.Statement;
import miniJava.visitor.miniJava.StringConstant;
import miniJava.visitor.miniJava.StringTypeRef;
import miniJava.visitor.miniJava.StringValue;
import miniJava.visitor.miniJava.Super;
import miniJava.visitor.miniJava.Superior;
import miniJava.visitor.miniJava.SuperiorOrEqual;
import miniJava.visitor.miniJava.Symbol;
import miniJava.visitor.miniJava.SymbolBinding;
import miniJava.visitor.miniJava.SymbolRef;
import miniJava.visitor.miniJava.This;
import miniJava.visitor.miniJava.TypeDeclaration;
import miniJava.visitor.miniJava.TypeRef;
import miniJava.visitor.miniJava.TypedDeclaration;
import miniJava.visitor.miniJava.Value;
import miniJava.visitor.miniJava.VariableDeclaration;
import miniJava.visitor.miniJava.VoidTypeRef;
import miniJava.visitor.miniJava.WhileStatement;
import miniJava.visitor.miniJava.WriteImage;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EDataType;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.impl.EFactoryImpl;
import org.eclipse.emf.ecore.plugin.EcorePlugin;

public class MiniJavaFactoryImpl extends EFactoryImpl implements MiniJavaFactory {
	public MiniJavaFactoryImpl() {
		super();
	}

	public static MiniJavaFactory init() {
		try {
			MiniJavaFactory theMiniJavaFactory = (MiniJavaFactory) EPackage.Registry.INSTANCE.getEFactory(MiniJavaPackage.eNS_URI);
			if (theMiniJavaFactory != null) {
				return theMiniJavaFactory;
			}
		} catch (Exception exception) {
			EcorePlugin.INSTANCE.log(exception);
		}
		return new MiniJavaFactoryImpl();
	}

	@Override
	public EObject create(EClass eClass) {
		switch (eClass.getClassifierID()) {
			case MiniJavaPackage.PROGRAM :
				return createProgram();
			case MiniJavaPackage.IMPORT :
				return createImport();
			case MiniJavaPackage.TYPE_DECLARATION :
				return createTypeDeclaration();
			case MiniJavaPackage.CLAZZ :
				return createClazz();
			case MiniJavaPackage.INTERFACE :
				return createInterface();
			case MiniJavaPackage.MEMBER :
				return createMember();
			case MiniJavaPackage.METHOD :
				return createMethod();
			case MiniJavaPackage.PARAMETER :
				return createParameter();
			case MiniJavaPackage.FIELD :
				return createField();
			case MiniJavaPackage.BLOCK :
				return createBlock();
			case MiniJavaPackage.STATEMENT :
				return createStatement();
			case MiniJavaPackage.PRINT_STATEMENT :
				return createPrintStatement();
			case MiniJavaPackage.LOAD_IMAGE :
				return createLoadImage();
			case MiniJavaPackage.WRITE_IMAGE :
				return createWriteImage();
			case MiniJavaPackage.RETURN :
				return createReturn();
			case MiniJavaPackage.IF_STATEMENT :
				return createIfStatement();
			case MiniJavaPackage.WHILE_STATEMENT :
				return createWhileStatement();
			case MiniJavaPackage.FOR_STATEMENT :
				return createForStatement();
			case MiniJavaPackage.TYPE_REF :
				return createTypeRef();
			case MiniJavaPackage.SINGLE_TYPE_REF :
				return createSingleTypeRef();
			case MiniJavaPackage.CLASS_REF :
				return createClassRef();
			case MiniJavaPackage.NAMED_ELEMENT :
				return createNamedElement();
			case MiniJavaPackage.TYPED_DECLARATION :
				return createTypedDeclaration();
			case MiniJavaPackage.SYMBOL :
				return createSymbol();
			case MiniJavaPackage.VARIABLE_DECLARATION :
				return createVariableDeclaration();
			case MiniJavaPackage.ASSIGNMENT :
				return createAssignment();
			case MiniJavaPackage.ASSIGNEE :
				return createAssignee();
			case MiniJavaPackage.EXPRESSION :
				return createExpression();
			case MiniJavaPackage.ARRAY_TYPE_REF :
				return createArrayTypeRef();
			case MiniJavaPackage.INTEGER_TYPE_REF :
				return createIntegerTypeRef();
			case MiniJavaPackage.BOOLEAN_TYPE_REF :
				return createBooleanTypeRef();
			case MiniJavaPackage.STRING_TYPE_REF :
				return createStringTypeRef();
			case MiniJavaPackage.VOID_TYPE_REF :
				return createVoidTypeRef();
			case MiniJavaPackage.OR :
				return createOr();
			case MiniJavaPackage.AND :
				return createAnd();
			case MiniJavaPackage.EQUALITY :
				return createEquality();
			case MiniJavaPackage.INEQUALITY :
				return createInequality();
			case MiniJavaPackage.SUPERIOR_OR_EQUAL :
				return createSuperiorOrEqual();
			case MiniJavaPackage.INFERIOR_OR_EQUAL :
				return createInferiorOrEqual();
			case MiniJavaPackage.SUPERIOR :
				return createSuperior();
			case MiniJavaPackage.INFERIOR :
				return createInferior();
			case MiniJavaPackage.PLUS :
				return createPlus();
			case MiniJavaPackage.MINUS :
				return createMinus();
			case MiniJavaPackage.MULTIPLICATION :
				return createMultiplication();
			case MiniJavaPackage.DIVISION :
				return createDivision();
			case MiniJavaPackage.ARRAY_ACCESS :
				return createArrayAccess();
			case MiniJavaPackage.ARRAY_LENGTH :
				return createArrayLength();
			case MiniJavaPackage.NOT :
				return createNot();
			case MiniJavaPackage.NEG :
				return createNeg();
			case MiniJavaPackage.FIELD_ACCESS :
				return createFieldAccess();
			case MiniJavaPackage.METHOD_CALL :
				return createMethodCall();
			case MiniJavaPackage.STRING_CONSTANT :
				return createStringConstant();
			case MiniJavaPackage.INT_CONSTANT :
				return createIntConstant();
			case MiniJavaPackage.BOOL_CONSTANT :
				return createBoolConstant();
			case MiniJavaPackage.THIS :
				return createThis();
			case MiniJavaPackage.SUPER :
				return createSuper();
			case MiniJavaPackage.NULL :
				return createNull();
			case MiniJavaPackage.NEW_OBJECT :
				return createNewObject();
			case MiniJavaPackage.NEW_ARRAY :
				return createNewArray();
			case MiniJavaPackage.SYMBOL_REF :
				return createSymbolRef();
			case MiniJavaPackage.CONTEXT :
				return createContext();
			case MiniJavaPackage.VALUE :
				return createValue();
			case MiniJavaPackage.INTEGER_VALUE :
				return createIntegerValue();
			case MiniJavaPackage.SYMBOL_BINDING :
				return createSymbolBinding();
			case MiniJavaPackage.FIELD_BINDING :
				return createFieldBinding();
			case MiniJavaPackage.STRING_VALUE :
				return createStringValue();
			case MiniJavaPackage.BOOLEAN_VALUE :
				return createBooleanValue();
			case MiniJavaPackage.OUTPUT_STREAM :
				return createOutputStream();
			case MiniJavaPackage.STATE :
				return createState();
			case MiniJavaPackage.FRAME :
				return createFrame();
			case MiniJavaPackage.NULL_VALUE :
				return createNullValue();
			case MiniJavaPackage.NEW_CALL :
				return createNewCall();
			case MiniJavaPackage.METHOD_CALL2 :
				return createMethodCall2();
			case MiniJavaPackage.OBJECT_INSTANCE :
				return createObjectInstance();
			case MiniJavaPackage.ARRAY_INSTANCE :
				return createArrayInstance();
			case MiniJavaPackage.OBJECT_REF_VALUE :
				return createObjectRefValue();
			case MiniJavaPackage.ARRAY_REF_VALUE :
				return createArrayRefValue();
			case MiniJavaPackage.SYMBOL_TO_SYMBOL_BINDING_MAP :
				return (EObject) createSymbolToSymbolBindingMap();
			case MiniJavaPackage.CLAZZ_TO_METHOD_MAP :
				return (EObject) createClazzToMethodMap();
			case MiniJavaPackage.MODULO :
				return createModulo();
			case MiniJavaPackage.SQRT :
				return createSqrt();
			default :
				throw new IllegalArgumentException("The class '" + eClass.getName() + "' is not a valid classifier");
		}
	}

	@Override
	public Object createFromString(EDataType eDataType, String initialValue) {
		switch (eDataType.getClassifierID()) {
			case MiniJavaPackage.ACCESS_LEVEL :
				return createAccessLevelFromString(eDataType, initialValue);
			default :
				throw new IllegalArgumentException("The datatype '" + eDataType.getName() + "' is not a valid classifier");
		}
	}

	@Override
	public String convertToString(EDataType eDataType, Object instanceValue) {
		switch (eDataType.getClassifierID()) {
			case MiniJavaPackage.ACCESS_LEVEL :
				return convertAccessLevelToString(eDataType, instanceValue);
			default :
				throw new IllegalArgumentException("The datatype '" + eDataType.getName() + "' is not a valid classifier");
		}
	}

	public Program createProgram() {
		ProgramImpl program = new ProgramImpl();
		return program;
	}

	public Import createImport() {
		ImportImpl import_ = new ImportImpl();
		return import_;
	}

	public TypeDeclaration createTypeDeclaration() {
		TypeDeclarationImpl typeDeclaration = new TypeDeclarationImpl();
		return typeDeclaration;
	}

	public Clazz createClazz() {
		ClazzImpl clazz = new ClazzImpl();
		return clazz;
	}

	public Interface createInterface() {
		InterfaceImpl interface_ = new InterfaceImpl();
		return interface_;
	}

	public Member createMember() {
		MemberImpl member = new MemberImpl();
		return member;
	}

	public Method createMethod() {
		MethodImpl method = new MethodImpl();
		return method;
	}

	public Parameter createParameter() {
		ParameterImpl parameter = new ParameterImpl();
		return parameter;
	}

	public Field createField() {
		FieldImpl field = new FieldImpl();
		return field;
	}

	public Block createBlock() {
		BlockImpl block = new BlockImpl();
		return block;
	}

	public Statement createStatement() {
		StatementImpl statement = new StatementImpl();
		return statement;
	}

	public PrintStatement createPrintStatement() {
		PrintStatementImpl printStatement = new PrintStatementImpl();
		return printStatement;
	}

	public LoadImage createLoadImage() {
		LoadImageImpl loadImage = new LoadImageImpl();
		return loadImage;
	}

	public WriteImage createWriteImage() {
		WriteImageImpl writeImage = new WriteImageImpl();
		return writeImage;
	}

	public Return createReturn() {
		ReturnImpl return_ = new ReturnImpl();
		return return_;
	}

	public IfStatement createIfStatement() {
		IfStatementImpl ifStatement = new IfStatementImpl();
		return ifStatement;
	}

	public WhileStatement createWhileStatement() {
		WhileStatementImpl whileStatement = new WhileStatementImpl();
		return whileStatement;
	}

	public ForStatement createForStatement() {
		ForStatementImpl forStatement = new ForStatementImpl();
		return forStatement;
	}

	public TypeRef createTypeRef() {
		TypeRefImpl typeRef = new TypeRefImpl();
		return typeRef;
	}

	public SingleTypeRef createSingleTypeRef() {
		SingleTypeRefImpl singleTypeRef = new SingleTypeRefImpl();
		return singleTypeRef;
	}

	public ClassRef createClassRef() {
		ClassRefImpl classRef = new ClassRefImpl();
		return classRef;
	}

	public NamedElement createNamedElement() {
		NamedElementImpl namedElement = new NamedElementImpl();
		return namedElement;
	}

	public TypedDeclaration createTypedDeclaration() {
		TypedDeclarationImpl typedDeclaration = new TypedDeclarationImpl();
		return typedDeclaration;
	}

	public Symbol createSymbol() {
		SymbolImpl symbol = new SymbolImpl();
		return symbol;
	}

	public VariableDeclaration createVariableDeclaration() {
		VariableDeclarationImpl variableDeclaration = new VariableDeclarationImpl();
		return variableDeclaration;
	}

	public Assignment createAssignment() {
		AssignmentImpl assignment = new AssignmentImpl();
		return assignment;
	}

	public Assignee createAssignee() {
		AssigneeImpl assignee = new AssigneeImpl();
		return assignee;
	}

	public Expression createExpression() {
		ExpressionImpl expression = new ExpressionImpl();
		return expression;
	}

	public ArrayTypeRef createArrayTypeRef() {
		ArrayTypeRefImpl arrayTypeRef = new ArrayTypeRefImpl();
		return arrayTypeRef;
	}

	public IntegerTypeRef createIntegerTypeRef() {
		IntegerTypeRefImpl integerTypeRef = new IntegerTypeRefImpl();
		return integerTypeRef;
	}

	public BooleanTypeRef createBooleanTypeRef() {
		BooleanTypeRefImpl booleanTypeRef = new BooleanTypeRefImpl();
		return booleanTypeRef;
	}

	public StringTypeRef createStringTypeRef() {
		StringTypeRefImpl stringTypeRef = new StringTypeRefImpl();
		return stringTypeRef;
	}

	public VoidTypeRef createVoidTypeRef() {
		VoidTypeRefImpl voidTypeRef = new VoidTypeRefImpl();
		return voidTypeRef;
	}

	public Or createOr() {
		OrImpl or = new OrImpl();
		return or;
	}

	public And createAnd() {
		AndImpl and = new AndImpl();
		return and;
	}

	public Equality createEquality() {
		EqualityImpl equality = new EqualityImpl();
		return equality;
	}

	public Inequality createInequality() {
		InequalityImpl inequality = new InequalityImpl();
		return inequality;
	}

	public SuperiorOrEqual createSuperiorOrEqual() {
		SuperiorOrEqualImpl superiorOrEqual = new SuperiorOrEqualImpl();
		return superiorOrEqual;
	}

	public InferiorOrEqual createInferiorOrEqual() {
		InferiorOrEqualImpl inferiorOrEqual = new InferiorOrEqualImpl();
		return inferiorOrEqual;
	}

	public Superior createSuperior() {
		SuperiorImpl superior = new SuperiorImpl();
		return superior;
	}

	public Inferior createInferior() {
		InferiorImpl inferior = new InferiorImpl();
		return inferior;
	}

	public Plus createPlus() {
		PlusImpl plus = new PlusImpl();
		return plus;
	}

	public Minus createMinus() {
		MinusImpl minus = new MinusImpl();
		return minus;
	}

	public Multiplication createMultiplication() {
		MultiplicationImpl multiplication = new MultiplicationImpl();
		return multiplication;
	}

	public Division createDivision() {
		DivisionImpl division = new DivisionImpl();
		return division;
	}

	public ArrayAccess createArrayAccess() {
		ArrayAccessImpl arrayAccess = new ArrayAccessImpl();
		return arrayAccess;
	}

	public ArrayLength createArrayLength() {
		ArrayLengthImpl arrayLength = new ArrayLengthImpl();
		return arrayLength;
	}

	public Not createNot() {
		NotImpl not = new NotImpl();
		return not;
	}

	public Neg createNeg() {
		NegImpl neg = new NegImpl();
		return neg;
	}

	public FieldAccess createFieldAccess() {
		FieldAccessImpl fieldAccess = new FieldAccessImpl();
		return fieldAccess;
	}

	public MethodCall createMethodCall() {
		MethodCallImpl methodCall = new MethodCallImpl();
		return methodCall;
	}

	public StringConstant createStringConstant() {
		StringConstantImpl stringConstant = new StringConstantImpl();
		return stringConstant;
	}

	public IntConstant createIntConstant() {
		IntConstantImpl intConstant = new IntConstantImpl();
		return intConstant;
	}

	public BoolConstant createBoolConstant() {
		BoolConstantImpl boolConstant = new BoolConstantImpl();
		return boolConstant;
	}

	public This createThis() {
		ThisImpl this_ = new ThisImpl();
		return this_;
	}

	public Super createSuper() {
		SuperImpl super_ = new SuperImpl();
		return super_;
	}

	public Null createNull() {
		NullImpl null_ = new NullImpl();
		return null_;
	}

	public NewObject createNewObject() {
		NewObjectImpl newObject = new NewObjectImpl();
		return newObject;
	}

	public NewArray createNewArray() {
		NewArrayImpl newArray = new NewArrayImpl();
		return newArray;
	}

	public SymbolRef createSymbolRef() {
		SymbolRefImpl symbolRef = new SymbolRefImpl();
		return symbolRef;
	}

	public Context createContext() {
		ContextImpl context = new ContextImpl();
		return context;
	}

	public Value createValue() {
		ValueImpl value = new ValueImpl();
		return value;
	}

	public IntegerValue createIntegerValue() {
		IntegerValueImpl integerValue = new IntegerValueImpl();
		return integerValue;
	}

	public SymbolBinding createSymbolBinding() {
		SymbolBindingImpl symbolBinding = new SymbolBindingImpl();
		return symbolBinding;
	}

	public FieldBinding createFieldBinding() {
		FieldBindingImpl fieldBinding = new FieldBindingImpl();
		return fieldBinding;
	}

	public StringValue createStringValue() {
		StringValueImpl stringValue = new StringValueImpl();
		return stringValue;
	}

	public BooleanValue createBooleanValue() {
		BooleanValueImpl booleanValue = new BooleanValueImpl();
		return booleanValue;
	}

	public OutputStream createOutputStream() {
		OutputStreamImpl outputStream = new OutputStreamImpl();
		return outputStream;
	}

	public State createState() {
		StateImpl state = new StateImpl();
		return state;
	}

	public Frame createFrame() {
		FrameImpl frame = new FrameImpl();
		return frame;
	}

	public NullValue createNullValue() {
		NullValueImpl nullValue = new NullValueImpl();
		return nullValue;
	}

	public NewCall createNewCall() {
		NewCallImpl newCall = new NewCallImpl();
		return newCall;
	}

	public MethodCall2 createMethodCall2() {
		MethodCall2Impl methodCall2 = new MethodCall2Impl();
		return methodCall2;
	}

	public ObjectInstance createObjectInstance() {
		ObjectInstanceImpl objectInstance = new ObjectInstanceImpl();
		return objectInstance;
	}

	public ArrayInstance createArrayInstance() {
		ArrayInstanceImpl arrayInstance = new ArrayInstanceImpl();
		return arrayInstance;
	}

	public ObjectRefValue createObjectRefValue() {
		ObjectRefValueImpl objectRefValue = new ObjectRefValueImpl();
		return objectRefValue;
	}

	public ArrayRefValue createArrayRefValue() {
		ArrayRefValueImpl arrayRefValue = new ArrayRefValueImpl();
		return arrayRefValue;
	}

	public Map.Entry<Symbol, SymbolBinding> createSymbolToSymbolBindingMap() {
		SymbolToSymbolBindingMapImpl symbolToSymbolBindingMap = new SymbolToSymbolBindingMapImpl();
		return symbolToSymbolBindingMap;
	}

	public Map.Entry<Clazz, Method> createClazzToMethodMap() {
		ClazzToMethodMapImpl clazzToMethodMap = new ClazzToMethodMapImpl();
		return clazzToMethodMap;
	}

	public Modulo createModulo() {
		ModuloImpl modulo = new ModuloImpl();
		return modulo;
	}

	public Sqrt createSqrt() {
		SqrtImpl sqrt = new SqrtImpl();
		return sqrt;
	}

	public AccessLevel createAccessLevelFromString(EDataType eDataType, String initialValue) {
		AccessLevel result = AccessLevel.get(initialValue);
		if (result == null)
			throw new IllegalArgumentException(
					"The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	public String convertAccessLevelToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	public MiniJavaPackage getMiniJavaPackage() {
		return (MiniJavaPackage) getEPackage();
	}

	@Deprecated
	public static MiniJavaPackage getPackage() {
		return MiniJavaPackage.eINSTANCE;
	}
}
