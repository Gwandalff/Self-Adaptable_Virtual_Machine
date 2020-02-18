package miniJava.visitor.miniJava.impl;

import java.util.Map;
import java.util.Set;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EDataType;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.impl.EFactoryImpl;
import org.eclipse.emf.ecore.plugin.EcorePlugin;

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
import visitor.IDynamicModule;
import visitor.dynamicmodules.DynamicApproximateModule;

public class MiniJavaFactoryImpl extends EFactoryImpl implements MiniJavaFactory {
	
	static Set<IDynamicModule> modules;
	Set<IDynamicModule> buffer;
	
	public MiniJavaFactoryImpl() {
		super();
	}

	public static MiniJavaFactory init() {
		instaciateDynamicModules();
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
		int id = eClass.getClassifierID();
		
		buffer.clear();
		for (IDynamicModule dm : modules) {
			int[] ids = dm.targetedNodes();
			for (int i = 0; i < ids.length; i++) {
				if(ids[i] == id) buffer.add(dm);
			}
		}
		
		switch (id) {
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
		for (IDynamicModule dm : buffer) {
			program.attach(dm);
		}
		return program;
	}

	public Import createImport() {
		ImportImpl import_ = new ImportImpl();
		for (IDynamicModule dm : buffer) {
			import_.attach(dm);
		}
		return import_;
	}

	public TypeDeclaration createTypeDeclaration() {
		TypeDeclarationImpl typeDeclaration = new TypeDeclarationImpl();
		for (IDynamicModule dm : buffer) {
			typeDeclaration.attach(dm);
		}
		return typeDeclaration;
	}

	public Clazz createClazz() {
		ClazzImpl clazz = new ClazzImpl();
		for (IDynamicModule dm : buffer) {
			clazz.attach(dm);
		}
		return clazz;
	}

	public Interface createInterface() {
		InterfaceImpl interface_ = new InterfaceImpl();
		for (IDynamicModule dm : buffer) {
			interface_.attach(dm);
		}
		return interface_;
	}

	public Member createMember() {
		MemberImpl member = new MemberImpl();
		for (IDynamicModule dm : buffer) {
			member.attach(dm);
		}
		return member;
	}

	public Method createMethod() {
		MethodImpl method = new MethodImpl();
		for (IDynamicModule dm : buffer) {
			method.attach(dm);
		}
		return method;
	}

	public Parameter createParameter() {
		ParameterImpl parameter = new ParameterImpl();
		for (IDynamicModule dm : buffer) {
			parameter.attach(dm);
		}
		return parameter;
	}

	public Field createField() {
		FieldImpl field = new FieldImpl();
		for (IDynamicModule dm : buffer) {
			field.attach(dm);
		}
		return field;
	}

	public Block createBlock() {
		BlockImpl block = new BlockImpl();
		for (IDynamicModule dm : buffer) {
			block.attach(dm);
		}
		return block;
	}

	public Statement createStatement() {
		StatementImpl statement = new StatementImpl();
		for (IDynamicModule dm : buffer) {
			statement.attach(dm);
		}
		return statement;
	}

	public PrintStatement createPrintStatement() {
		PrintStatementImpl printStatement = new PrintStatementImpl();
		for (IDynamicModule dm : buffer) {
			printStatement.attach(dm);
		}
		return printStatement;
	}

	public Return createReturn() {
		ReturnImpl return_ = new ReturnImpl();
		for (IDynamicModule dm : buffer) {
			return_.attach(dm);
		}
		return return_;
	}

	public IfStatement createIfStatement() {
		IfStatementImpl ifStatement = new IfStatementImpl();
		for (IDynamicModule dm : buffer) {
			ifStatement.attach(dm);
		}
		return ifStatement;
	}

	public WhileStatement createWhileStatement() {
		WhileStatementImpl whileStatement = new WhileStatementImpl();
		for (IDynamicModule dm : buffer) {
			whileStatement.attach(dm);
		}
		return whileStatement;
	}

	public ForStatement createForStatement() {
		ForStatementImpl forStatement = new ForStatementImpl();
		for (IDynamicModule dm : buffer) {
			forStatement.attach(dm);
		}
		return forStatement;
	}

	public TypeRef createTypeRef() {
		TypeRefImpl typeRef = new TypeRefImpl();
		for (IDynamicModule dm : buffer) {
			typeRef.attach(dm);
		}
		return typeRef;
	}

	public SingleTypeRef createSingleTypeRef() {
		SingleTypeRefImpl singleTypeRef = new SingleTypeRefImpl();
		for (IDynamicModule dm : buffer) {
			singleTypeRef.attach(dm);
		}
		return singleTypeRef;
	}

	public ClassRef createClassRef() {
		ClassRefImpl classRef = new ClassRefImpl();
		for (IDynamicModule dm : buffer) {
			classRef.attach(dm);
		}
		return classRef;
	}

	public NamedElement createNamedElement() {
		NamedElementImpl namedElement = new NamedElementImpl();
		for (IDynamicModule dm : buffer) {
			namedElement.attach(dm);
		}
		return namedElement;
	}

	public TypedDeclaration createTypedDeclaration() {
		TypedDeclarationImpl typedDeclaration = new TypedDeclarationImpl();
		for (IDynamicModule dm : buffer) {
			typedDeclaration.attach(dm);
		}
		return typedDeclaration;
	}

	public Symbol createSymbol() {
		SymbolImpl symbol = new SymbolImpl();
		for (IDynamicModule dm : buffer) {
			symbol.attach(dm);
		}
		return symbol;
	}

	public VariableDeclaration createVariableDeclaration() {
		VariableDeclarationImpl variableDeclaration = new VariableDeclarationImpl();
		for (IDynamicModule dm : buffer) {
			variableDeclaration.attach(dm);
		}
		return variableDeclaration;
	}

	public Assignment createAssignment() {
		AssignmentImpl assignment = new AssignmentImpl();
		for (IDynamicModule dm : buffer) {
			assignment.attach(dm);
		}
		return assignment;
	}

	public Assignee createAssignee() {
		AssigneeImpl assignee = new AssigneeImpl();
		for (IDynamicModule dm : buffer) {
			assignee.attach(dm);
		}
		return assignee;
	}

	public Expression createExpression() {
		ExpressionImpl expression = new ExpressionImpl();
		for (IDynamicModule dm : buffer) {
			expression.attach(dm);
		}
		return expression;
	}

	public ArrayTypeRef createArrayTypeRef() {
		ArrayTypeRefImpl arrayTypeRef = new ArrayTypeRefImpl();
		for (IDynamicModule dm : buffer) {
			arrayTypeRef.attach(dm);
		}
		return arrayTypeRef;
	}

	public IntegerTypeRef createIntegerTypeRef() {
		IntegerTypeRefImpl integerTypeRef = new IntegerTypeRefImpl();
		for (IDynamicModule dm : buffer) {
			integerTypeRef.attach(dm);
		}
		return integerTypeRef;
	}

	public BooleanTypeRef createBooleanTypeRef() {
		BooleanTypeRefImpl booleanTypeRef = new BooleanTypeRefImpl();
		for (IDynamicModule dm : buffer) {
			booleanTypeRef.attach(dm);
		}
		return booleanTypeRef;
	}

	public StringTypeRef createStringTypeRef() {
		StringTypeRefImpl stringTypeRef = new StringTypeRefImpl();
		for (IDynamicModule dm : buffer) {
			stringTypeRef.attach(dm);
		}
		return stringTypeRef;
	}

	public VoidTypeRef createVoidTypeRef() {
		VoidTypeRefImpl voidTypeRef = new VoidTypeRefImpl();
		for (IDynamicModule dm : buffer) {
			voidTypeRef.attach(dm);
		}
		return voidTypeRef;
	}

	public Or createOr() {
		OrImpl or = new OrImpl();
		for (IDynamicModule dm : buffer) {
			or.attach(dm);
		}
		return or;
	}

	public And createAnd() {
		AndImpl and = new AndImpl();
		for (IDynamicModule dm : buffer) {
			and.attach(dm);
		}
		return and;
	}

	public Equality createEquality() {
		EqualityImpl equality = new EqualityImpl();
		for (IDynamicModule dm : buffer) {
			equality.attach(dm);
		}
		return equality;
	}

	public Inequality createInequality() {
		InequalityImpl inequality = new InequalityImpl();
		for (IDynamicModule dm : buffer) {
			inequality.attach(dm);
		}
		return inequality;
	}

	public SuperiorOrEqual createSuperiorOrEqual() {
		SuperiorOrEqualImpl superiorOrEqual = new SuperiorOrEqualImpl();
		for (IDynamicModule dm : buffer) {
			superiorOrEqual.attach(dm);
		}
		return superiorOrEqual;
	}

	public InferiorOrEqual createInferiorOrEqual() {
		InferiorOrEqualImpl inferiorOrEqual = new InferiorOrEqualImpl();
		for (IDynamicModule dm : buffer) {
			inferiorOrEqual.attach(dm);
		}
		return inferiorOrEqual;
	}

	public Superior createSuperior() {
		SuperiorImpl superior = new SuperiorImpl();
		for (IDynamicModule dm : buffer) {
			superior.attach(dm);
		}
		return superior;
	}

	public Inferior createInferior() {
		InferiorImpl inferior = new InferiorImpl();
		for (IDynamicModule dm : buffer) {
			inferior.attach(dm);
		}
		return inferior;
	}

	public Plus createPlus() {
		PlusImpl plus = new PlusImpl();
		for (IDynamicModule dm : buffer) {
			plus.attach(dm);
		}
		return plus;
	}

	public Minus createMinus() {
		MinusImpl minus = new MinusImpl();
		for (IDynamicModule dm : buffer) {
			minus.attach(dm);
		}
		return minus;
	}

	public Multiplication createMultiplication() {
		MultiplicationImpl multiplication = new MultiplicationImpl();
		for (IDynamicModule dm : buffer) {
			multiplication.attach(dm);
		}
		return multiplication;
	}

	public Division createDivision() {
		DivisionImpl division = new DivisionImpl();
		for (IDynamicModule dm : buffer) {
			division.attach(dm);
		}
		return division;
	}

	public ArrayAccess createArrayAccess() {
		ArrayAccessImpl arrayAccess = new ArrayAccessImpl();
		for (IDynamicModule dm : buffer) {
			arrayAccess.attach(dm);
		}
		return arrayAccess;
	}

	public ArrayLength createArrayLength() {
		ArrayLengthImpl arrayLength = new ArrayLengthImpl();
		for (IDynamicModule dm : buffer) {
			arrayLength.attach(dm);
		}
		return arrayLength;
	}

	public Not createNot() {
		NotImpl not = new NotImpl();
		for (IDynamicModule dm : buffer) {
			not.attach(dm);
		}
		return not;
	}

	public Neg createNeg() {
		NegImpl neg = new NegImpl();
		for (IDynamicModule dm : buffer) {
			neg.attach(dm);
		}
		return neg;
	}

	public FieldAccess createFieldAccess() {
		FieldAccessImpl fieldAccess = new FieldAccessImpl();
		for (IDynamicModule dm : buffer) {
			fieldAccess.attach(dm);
		}
		return fieldAccess;
	}

	public MethodCall createMethodCall() {
		MethodCallImpl methodCall = new MethodCallImpl();
		for (IDynamicModule dm : buffer) {
			methodCall.attach(dm);
		}
		return methodCall;
	}

	public StringConstant createStringConstant() {
		StringConstantImpl stringConstant = new StringConstantImpl();
		for (IDynamicModule dm : buffer) {
			stringConstant.attach(dm);
		}
		return stringConstant;
	}

	public IntConstant createIntConstant() {
		IntConstantImpl intConstant = new IntConstantImpl();
		for (IDynamicModule dm : buffer) {
			intConstant.attach(dm);
		}
		return intConstant;
	}

	public BoolConstant createBoolConstant() {
		BoolConstantImpl boolConstant = new BoolConstantImpl();
		for (IDynamicModule dm : buffer) {
			boolConstant.attach(dm);
		}
		return boolConstant;
	}

	public This createThis() {
		ThisImpl this_ = new ThisImpl();
		for (IDynamicModule dm : buffer) {
			this_.attach(dm);
		}
		return this_;
	}

	public Super createSuper() {
		SuperImpl super_ = new SuperImpl();
		for (IDynamicModule dm : buffer) {
			super_.attach(dm);
		}
		return super_;
	}

	public Null createNull() {
		NullImpl null_ = new NullImpl();
		for (IDynamicModule dm : buffer) {
			null_.attach(dm);
		}
		return null_;
	}

	public NewObject createNewObject() {
		NewObjectImpl newObject = new NewObjectImpl();
		for (IDynamicModule dm : buffer) {
			newObject.attach(dm);
		}
		return newObject;
	}

	public NewArray createNewArray() {
		NewArrayImpl newArray = new NewArrayImpl();
		for (IDynamicModule dm : buffer) {
			newArray.attach(dm);
		}
		return newArray;
	}

	public SymbolRef createSymbolRef() {
		SymbolRefImpl symbolRef = new SymbolRefImpl();
		for (IDynamicModule dm : buffer) {
			symbolRef.attach(dm);
		}
		return symbolRef;
	}

	public Context createContext() {
		ContextImpl context = new ContextImpl();
		for (IDynamicModule dm : buffer) {
			context.attach(dm);
		}
		return context;
	}

	public Value createValue() {
		ValueImpl value = new ValueImpl();
		for (IDynamicModule dm : buffer) {
			value.attach(dm);
		}
		return value;
	}

	public IntegerValue createIntegerValue() {
		IntegerValueImpl integerValue = new IntegerValueImpl();
		for (IDynamicModule dm : buffer) {
			integerValue.attach(dm);
		}
		return integerValue;
	}

	public SymbolBinding createSymbolBinding() {
		SymbolBindingImpl symbolBinding = new SymbolBindingImpl();
		for (IDynamicModule dm : buffer) {
			symbolBinding.attach(dm);
		}
		return symbolBinding;
	}

	public FieldBinding createFieldBinding() {
		FieldBindingImpl fieldBinding = new FieldBindingImpl();
		for (IDynamicModule dm : buffer) {
			fieldBinding.attach(dm);
		}
		return fieldBinding;
	}

	public StringValue createStringValue() {
		StringValueImpl stringValue = new StringValueImpl();
		for (IDynamicModule dm : buffer) {
			stringValue.attach(dm);
		}
		return stringValue;
	}

	public BooleanValue createBooleanValue() {
		BooleanValueImpl booleanValue = new BooleanValueImpl();
		for (IDynamicModule dm : buffer) {
			booleanValue.attach(dm);
		}
		return booleanValue;
	}

	public OutputStream createOutputStream() {
		OutputStreamImpl outputStream = new OutputStreamImpl();
		for (IDynamicModule dm : buffer) {
			outputStream.attach(dm);
		}
		return outputStream;
	}

	public State createState() {
		StateImpl state = new StateImpl();
		for (IDynamicModule dm : buffer) {
			state.attach(dm);
		}
		return state;
	}

	public Frame createFrame() {
		FrameImpl frame = new FrameImpl();
		for (IDynamicModule dm : buffer) {
			frame.attach(dm);
		}
		return frame;
	}

	public NullValue createNullValue() {
		NullValueImpl nullValue = new NullValueImpl();
		for (IDynamicModule dm : buffer) {
			nullValue.attach(dm);
		}
		return nullValue;
	}

	public NewCall createNewCall() {
		NewCallImpl newCall = new NewCallImpl();
		for (IDynamicModule dm : buffer) {
			newCall.attach(dm);
		}
		return newCall;
	}

	public MethodCall2 createMethodCall2() {
		MethodCall2Impl methodCall2 = new MethodCall2Impl();
		for (IDynamicModule dm : buffer) {
			methodCall2.attach(dm);
		}
		return methodCall2;
	}

	public ObjectInstance createObjectInstance() {
		ObjectInstanceImpl objectInstance = new ObjectInstanceImpl();
		for (IDynamicModule dm : buffer) {
			objectInstance.attach(dm);
		}
		return objectInstance;
	}

	public ArrayInstance createArrayInstance() {
		ArrayInstanceImpl arrayInstance = new ArrayInstanceImpl();
		for (IDynamicModule dm : buffer) {
			arrayInstance.attach(dm);
		}
		return arrayInstance;
	}

	public ObjectRefValue createObjectRefValue() {
		ObjectRefValueImpl objectRefValue = new ObjectRefValueImpl();
		for (IDynamicModule dm : buffer) {
			objectRefValue.attach(dm);
		}
		return objectRefValue;
	}

	public ArrayRefValue createArrayRefValue() {
		ArrayRefValueImpl arrayRefValue = new ArrayRefValueImpl();
		for (IDynamicModule dm : buffer) {
			arrayRefValue.attach(dm);
		}
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
		for (IDynamicModule dm : buffer) {
			modulo.attach(dm);
		}
		return modulo;
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
	
	private static void instaciateDynamicModules() {
		modules.add(new DynamicApproximateModule());
	}
}
