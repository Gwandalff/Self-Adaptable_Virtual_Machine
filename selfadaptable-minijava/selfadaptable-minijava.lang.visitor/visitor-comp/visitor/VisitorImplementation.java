package visitor;

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
import miniJava.visitor.miniJava.ClazzToMethodMap;
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
import miniJava.visitor.miniJava.SymbolToSymbolBindingMap;
import miniJava.visitor.miniJava.This;
import miniJava.visitor.miniJava.TypeDeclaration;
import miniJava.visitor.miniJava.TypeRef;
import miniJava.visitor.miniJava.TypedDeclaration;
import miniJava.visitor.miniJava.Value;
import miniJava.visitor.miniJava.VariableDeclaration;
import miniJava.visitor.miniJava.VoidTypeRef;
import miniJava.visitor.miniJava.WhileStatement;
import miniJava.visitor.miniJava.WriteImage;
import visitor.operation.miniJava.AndOperation;
import visitor.operation.miniJava.ArrayAccessOperation;
import visitor.operation.miniJava.ArrayInstanceOperation;
import visitor.operation.miniJava.ArrayLengthOperation;
import visitor.operation.miniJava.ArrayRefValueOperation;
import visitor.operation.miniJava.ArrayTypeRefOperation;
import visitor.operation.miniJava.AssigneeOperation;
import visitor.operation.miniJava.AssignmentOperation;
import visitor.operation.miniJava.BlockOperation;
import visitor.operation.miniJava.BoolConstantOperation;
import visitor.operation.miniJava.BooleanTypeRefOperation;
import visitor.operation.miniJava.BooleanValueOperation;
import visitor.operation.miniJava.ClassRefOperation;
import visitor.operation.miniJava.ClazzOperation;
import visitor.operation.miniJava.ClazzToMethodMapOperation;
import visitor.operation.miniJava.ContextOperation;
import visitor.operation.miniJava.DivisionOperation;
import visitor.operation.miniJava.EqualityOperation;
import visitor.operation.miniJava.ExpressionOperation;
import visitor.operation.miniJava.FieldAccessOperation;
import visitor.operation.miniJava.FieldBindingOperation;
import visitor.operation.miniJava.FieldOperation;
import visitor.operation.miniJava.ForStatementOperation;
import visitor.operation.miniJava.FrameOperation;
import visitor.operation.miniJava.IfStatementOperation;
import visitor.operation.miniJava.ImportOperation;
import visitor.operation.miniJava.InequalityOperation;
import visitor.operation.miniJava.InferiorOperation;
import visitor.operation.miniJava.InferiorOrEqualOperation;
import visitor.operation.miniJava.IntConstantOperation;
import visitor.operation.miniJava.IntegerTypeRefOperation;
import visitor.operation.miniJava.IntegerValueOperation;
import visitor.operation.miniJava.InterfaceOperation;
import visitor.operation.miniJava.LoadImageOperation;
import visitor.operation.miniJava.MemberOperation;
import visitor.operation.miniJava.MethodCall2Operation;
import visitor.operation.miniJava.MethodCallOperation;
import visitor.operation.miniJava.MethodOperation;
import visitor.operation.miniJava.MinusOperation;
import visitor.operation.miniJava.ModuloOperation;
import visitor.operation.miniJava.MultiplicationOperation;
import visitor.operation.miniJava.NamedElementOperation;
import visitor.operation.miniJava.NegOperation;
import visitor.operation.miniJava.NewArrayOperation;
import visitor.operation.miniJava.NewCallOperation;
import visitor.operation.miniJava.NewObjectOperation;
import visitor.operation.miniJava.NotOperation;
import visitor.operation.miniJava.NullOperation;
import visitor.operation.miniJava.NullValueOperation;
import visitor.operation.miniJava.ObjectInstanceOperation;
import visitor.operation.miniJava.ObjectRefValueOperation;
import visitor.operation.miniJava.OrOperation;
import visitor.operation.miniJava.OutputStreamOperation;
import visitor.operation.miniJava.ParameterOperation;
import visitor.operation.miniJava.PlusOperation;
import visitor.operation.miniJava.PrintStatementOperation;
import visitor.operation.miniJava.ProgramOperation;
import visitor.operation.miniJava.ReturnOperation;
import visitor.operation.miniJava.SingleTypeRefOperation;
import visitor.operation.miniJava.SqrtOperation;
import visitor.operation.miniJava.StateOperation;
import visitor.operation.miniJava.StatementOperation;
import visitor.operation.miniJava.StringConstantOperation;
import visitor.operation.miniJava.StringTypeRefOperation;
import visitor.operation.miniJava.StringValueOperation;
import visitor.operation.miniJava.SuperOperation;
import visitor.operation.miniJava.SuperiorOperation;
import visitor.operation.miniJava.SuperiorOrEqualOperation;
import visitor.operation.miniJava.SymbolBindingOperation;
import visitor.operation.miniJava.SymbolOperation;
import visitor.operation.miniJava.SymbolRefOperation;
import visitor.operation.miniJava.SymbolToSymbolBindingMapOperation;
import visitor.operation.miniJava.ThisOperation;
import visitor.operation.miniJava.TypeDeclarationOperation;
import visitor.operation.miniJava.TypeRefOperation;
import visitor.operation.miniJava.TypedDeclarationOperation;
import visitor.operation.miniJava.ValueOperation;
import visitor.operation.miniJava.VariableDeclarationOperation;
import visitor.operation.miniJava.VoidTypeRefOperation;
import visitor.operation.miniJava.WhileStatementOperation;
import visitor.operation.miniJava.WriteImageOperation;
import visitor.operation.miniJava.impl.AndOperationImpl;
import visitor.operation.miniJava.impl.ArrayAccessOperationImpl;
import visitor.operation.miniJava.impl.ArrayInstanceOperationImpl;
import visitor.operation.miniJava.impl.ArrayLengthOperationImpl;
import visitor.operation.miniJava.impl.ArrayRefValueOperationImpl;
import visitor.operation.miniJava.impl.ArrayTypeRefOperationImpl;
import visitor.operation.miniJava.impl.AssigneeOperationImpl;
import visitor.operation.miniJava.impl.AssignmentOperationImpl;
import visitor.operation.miniJava.impl.BlockOperationImpl;
import visitor.operation.miniJava.impl.BoolConstantOperationImpl;
import visitor.operation.miniJava.impl.BooleanTypeRefOperationImpl;
import visitor.operation.miniJava.impl.BooleanValueOperationImpl;
import visitor.operation.miniJava.impl.ClassRefOperationImpl;
import visitor.operation.miniJava.impl.ClazzOperationImpl;
import visitor.operation.miniJava.impl.ClazzToMethodMapOperationImpl;
import visitor.operation.miniJava.impl.ContextOperationImpl;
import visitor.operation.miniJava.impl.DivisionOperationImpl;
import visitor.operation.miniJava.impl.EqualityOperationImpl;
import visitor.operation.miniJava.impl.ExpressionOperationImpl;
import visitor.operation.miniJava.impl.FieldAccessOperationImpl;
import visitor.operation.miniJava.impl.FieldBindingOperationImpl;
import visitor.operation.miniJava.impl.FieldOperationImpl;
import visitor.operation.miniJava.impl.ForStatementOperationImpl;
import visitor.operation.miniJava.impl.FrameOperationImpl;
import visitor.operation.miniJava.impl.IfStatementOperationImpl;
import visitor.operation.miniJava.impl.ImportOperationImpl;
import visitor.operation.miniJava.impl.InequalityOperationImpl;
import visitor.operation.miniJava.impl.InferiorOperationImpl;
import visitor.operation.miniJava.impl.InferiorOrEqualOperationImpl;
import visitor.operation.miniJava.impl.IntConstantOperationImpl;
import visitor.operation.miniJava.impl.IntegerTypeRefOperationImpl;
import visitor.operation.miniJava.impl.IntegerValueOperationImpl;
import visitor.operation.miniJava.impl.InterfaceOperationImpl;
import visitor.operation.miniJava.impl.LoadImageOperationImpl;
import visitor.operation.miniJava.impl.MemberOperationImpl;
import visitor.operation.miniJava.impl.MethodCall2OperationImpl;
import visitor.operation.miniJava.impl.MethodCallOperationImpl;
import visitor.operation.miniJava.impl.MethodOperationImpl;
import visitor.operation.miniJava.impl.MinusOperationImpl;
import visitor.operation.miniJava.impl.ModuloOperationImpl;
import visitor.operation.miniJava.impl.MultiplicationOperationImpl;
import visitor.operation.miniJava.impl.NamedElementOperationImpl;
import visitor.operation.miniJava.impl.NegOperationImpl;
import visitor.operation.miniJava.impl.NewArrayOperationImpl;
import visitor.operation.miniJava.impl.NewCallOperationImpl;
import visitor.operation.miniJava.impl.NewObjectOperationImpl;
import visitor.operation.miniJava.impl.NotOperationImpl;
import visitor.operation.miniJava.impl.NullOperationImpl;
import visitor.operation.miniJava.impl.NullValueOperationImpl;
import visitor.operation.miniJava.impl.ObjectInstanceOperationImpl;
import visitor.operation.miniJava.impl.ObjectRefValueOperationImpl;
import visitor.operation.miniJava.impl.OrOperationImpl;
import visitor.operation.miniJava.impl.OutputStreamOperationImpl;
import visitor.operation.miniJava.impl.ParameterOperationImpl;
import visitor.operation.miniJava.impl.PlusOperationImpl;
import visitor.operation.miniJava.impl.PrintStatementOperationImpl;
import visitor.operation.miniJava.impl.ProgramOperationImpl;
import visitor.operation.miniJava.impl.ReturnOperationImpl;
import visitor.operation.miniJava.impl.SingleTypeRefOperationImpl;
import visitor.operation.miniJava.impl.SqrtOperationImpl;
import visitor.operation.miniJava.impl.StateOperationImpl;
import visitor.operation.miniJava.impl.StatementOperationImpl;
import visitor.operation.miniJava.impl.StringConstantOperationImpl;
import visitor.operation.miniJava.impl.StringTypeRefOperationImpl;
import visitor.operation.miniJava.impl.StringValueOperationImpl;
import visitor.operation.miniJava.impl.SuperOperationImpl;
import visitor.operation.miniJava.impl.SuperiorOperationImpl;
import visitor.operation.miniJava.impl.SuperiorOrEqualOperationImpl;
import visitor.operation.miniJava.impl.SymbolBindingOperationImpl;
import visitor.operation.miniJava.impl.SymbolOperationImpl;
import visitor.operation.miniJava.impl.SymbolRefOperationImpl;
import visitor.operation.miniJava.impl.SymbolToSymbolBindingMapOperationImpl;
import visitor.operation.miniJava.impl.ThisOperationImpl;
import visitor.operation.miniJava.impl.TypeDeclarationOperationImpl;
import visitor.operation.miniJava.impl.TypeRefOperationImpl;
import visitor.operation.miniJava.impl.TypedDeclarationOperationImpl;
import visitor.operation.miniJava.impl.ValueOperationImpl;
import visitor.operation.miniJava.impl.VariableDeclarationOperationImpl;
import visitor.operation.miniJava.impl.VoidTypeRefOperationImpl;
import visitor.operation.miniJava.impl.WhileStatementOperationImpl;
import visitor.operation.miniJava.impl.WriteImageOperationImpl;

public class VisitorImplementation implements VisitorInterface {
	public ProgramOperation visitminiJava__Program(Program it) {
		return new ProgramOperationImpl(it, this);
	}

	public ImportOperation visitminiJava__Import(Import it) {
		return new ImportOperationImpl(it, this);
	}

	public TypeDeclarationOperation visitminiJava__TypeDeclaration(TypeDeclaration it) {
		return new TypeDeclarationOperationImpl(it, this);
	}

	public ClazzOperation visitminiJava__Clazz(Clazz it) {
		return new ClazzOperationImpl(it, this);
	}

	public InterfaceOperation visitminiJava__Interface(Interface it) {
		return new InterfaceOperationImpl(it, this);
	}

	public MemberOperation visitminiJava__Member(Member it) {
		return new MemberOperationImpl(it, this);
	}

	public MethodOperation visitminiJava__Method(Method it) {
		return new MethodOperationImpl(it, this);
	}

	public ParameterOperation visitminiJava__Parameter(Parameter it) {
		return new ParameterOperationImpl(it, this);
	}

	public FieldOperation visitminiJava__Field(Field it) {
		return new FieldOperationImpl(it, this);
	}

	public BlockOperation visitminiJava__Block(Block it) {
		return new BlockOperationImpl(it, this);
	}

	public StatementOperation visitminiJava__Statement(Statement it) {
		return new StatementOperationImpl(it, this);
	}

	public PrintStatementOperation visitminiJava__PrintStatement(PrintStatement it) {
		return new PrintStatementOperationImpl(it, this);
	}

	public LoadImageOperation visitminiJava__LoadImage(LoadImage it) {
		return new LoadImageOperationImpl(it, this);
	}

	public WriteImageOperation visitminiJava__WriteImage(WriteImage it) {
		return new WriteImageOperationImpl(it, this);
	}

	public ReturnOperation visitminiJava__Return(Return it) {
		return new ReturnOperationImpl(it, this);
	}

	public IfStatementOperation visitminiJava__IfStatement(IfStatement it) {
		return new IfStatementOperationImpl(it, this);
	}

	public WhileStatementOperation visitminiJava__WhileStatement(WhileStatement it) {
		return new WhileStatementOperationImpl(it, this);
	}

	public ForStatementOperation visitminiJava__ForStatement(ForStatement it) {
		return new ForStatementOperationImpl(it, this);
	}

	public TypeRefOperation visitminiJava__TypeRef(TypeRef it) {
		return new TypeRefOperationImpl(it, this);
	}

	public SingleTypeRefOperation visitminiJava__SingleTypeRef(SingleTypeRef it) {
		return new SingleTypeRefOperationImpl(it, this);
	}

	public ClassRefOperation visitminiJava__ClassRef(ClassRef it) {
		return new ClassRefOperationImpl(it, this);
	}

	public NamedElementOperation visitminiJava__NamedElement(NamedElement it) {
		return new NamedElementOperationImpl(it, this);
	}

	public TypedDeclarationOperation visitminiJava__TypedDeclaration(TypedDeclaration it) {
		return new TypedDeclarationOperationImpl(it, this);
	}

	public SymbolOperation visitminiJava__Symbol(Symbol it) {
		return new SymbolOperationImpl(it, this);
	}

	public VariableDeclarationOperation visitminiJava__VariableDeclaration(VariableDeclaration it) {
		return new VariableDeclarationOperationImpl(it, this);
	}

	public AssignmentOperation visitminiJava__Assignment(Assignment it) {
		return new AssignmentOperationImpl(it, this);
	}

	public AssigneeOperation visitminiJava__Assignee(Assignee it) {
		return new AssigneeOperationImpl(it, this);
	}

	public ExpressionOperation visitminiJava__Expression(Expression it) {
		return new ExpressionOperationImpl(it, this);
	}

	public ArrayTypeRefOperation visitminiJava__ArrayTypeRef(ArrayTypeRef it) {
		return new ArrayTypeRefOperationImpl(it, this);
	}

	public IntegerTypeRefOperation visitminiJava__IntegerTypeRef(IntegerTypeRef it) {
		return new IntegerTypeRefOperationImpl(it, this);
	}

	public BooleanTypeRefOperation visitminiJava__BooleanTypeRef(BooleanTypeRef it) {
		return new BooleanTypeRefOperationImpl(it, this);
	}

	public StringTypeRefOperation visitminiJava__StringTypeRef(StringTypeRef it) {
		return new StringTypeRefOperationImpl(it, this);
	}

	public VoidTypeRefOperation visitminiJava__VoidTypeRef(VoidTypeRef it) {
		return new VoidTypeRefOperationImpl(it, this);
	}

	public OrOperation visitminiJava__Or(Or it) {
		return new OrOperationImpl(it, this);
	}

	public AndOperation visitminiJava__And(And it) {
		return new AndOperationImpl(it, this);
	}

	public EqualityOperation visitminiJava__Equality(Equality it) {
		return new EqualityOperationImpl(it, this);
	}

	public InequalityOperation visitminiJava__Inequality(Inequality it) {
		return new InequalityOperationImpl(it, this);
	}

	public SuperiorOrEqualOperation visitminiJava__SuperiorOrEqual(SuperiorOrEqual it) {
		return new SuperiorOrEqualOperationImpl(it, this);
	}

	public InferiorOrEqualOperation visitminiJava__InferiorOrEqual(InferiorOrEqual it) {
		return new InferiorOrEqualOperationImpl(it, this);
	}

	public SuperiorOperation visitminiJava__Superior(Superior it) {
		return new SuperiorOperationImpl(it, this);
	}

	public InferiorOperation visitminiJava__Inferior(Inferior it) {
		return new InferiorOperationImpl(it, this);
	}

	public PlusOperation visitminiJava__Plus(Plus it) {
		return new PlusOperationImpl(it, this);
	}

	public MinusOperation visitminiJava__Minus(Minus it) {
		return new MinusOperationImpl(it, this);
	}

	public MultiplicationOperation visitminiJava__Multiplication(Multiplication it) {
		return new MultiplicationOperationImpl(it, this);
	}

	public DivisionOperation visitminiJava__Division(Division it) {
		return new DivisionOperationImpl(it, this);
	}

	public ArrayAccessOperation visitminiJava__ArrayAccess(ArrayAccess it) {
		return new ArrayAccessOperationImpl(it, this);
	}

	public ArrayLengthOperation visitminiJava__ArrayLength(ArrayLength it) {
		return new ArrayLengthOperationImpl(it, this);
	}

	public NotOperation visitminiJava__Not(Not it) {
		return new NotOperationImpl(it, this);
	}

	public NegOperation visitminiJava__Neg(Neg it) {
		return new NegOperationImpl(it, this);
	}

	public FieldAccessOperation visitminiJava__FieldAccess(FieldAccess it) {
		return new FieldAccessOperationImpl(it, this);
	}

	public MethodCallOperation visitminiJava__MethodCall(MethodCall it) {
		return new MethodCallOperationImpl(it, this);
	}

	public StringConstantOperation visitminiJava__StringConstant(StringConstant it) {
		return new StringConstantOperationImpl(it, this);
	}

	public IntConstantOperation visitminiJava__IntConstant(IntConstant it) {
		return new IntConstantOperationImpl(it, this);
	}

	public BoolConstantOperation visitminiJava__BoolConstant(BoolConstant it) {
		return new BoolConstantOperationImpl(it, this);
	}

	public ThisOperation visitminiJava__This(This it) {
		return new ThisOperationImpl(it, this);
	}

	public SuperOperation visitminiJava__Super(Super it) {
		return new SuperOperationImpl(it, this);
	}

	public NullOperation visitminiJava__Null(Null it) {
		return new NullOperationImpl(it, this);
	}

	public NewObjectOperation visitminiJava__NewObject(NewObject it) {
		return new NewObjectOperationImpl(it, this);
	}

	public NewArrayOperation visitminiJava__NewArray(NewArray it) {
		return new NewArrayOperationImpl(it, this);
	}

	public SymbolRefOperation visitminiJava__SymbolRef(SymbolRef it) {
		return new SymbolRefOperationImpl(it, this);
	}

	public ContextOperation visitminiJava__Context(Context it) {
		return new ContextOperationImpl(it, this);
	}

	public ValueOperation visitminiJava__Value(Value it) {
		return new ValueOperationImpl(it, this);
	}

	public IntegerValueOperation visitminiJava__IntegerValue(IntegerValue it) {
		return new IntegerValueOperationImpl(it, this);
	}

	public SymbolBindingOperation visitminiJava__SymbolBinding(SymbolBinding it) {
		return new SymbolBindingOperationImpl(it, this);
	}

	public FieldBindingOperation visitminiJava__FieldBinding(FieldBinding it) {
		return new FieldBindingOperationImpl(it, this);
	}

	public StringValueOperation visitminiJava__StringValue(StringValue it) {
		return new StringValueOperationImpl(it, this);
	}

	public BooleanValueOperation visitminiJava__BooleanValue(BooleanValue it) {
		return new BooleanValueOperationImpl(it, this);
	}

	public OutputStreamOperation visitminiJava__OutputStream(OutputStream it) {
		return new OutputStreamOperationImpl(it, this);
	}

	public StateOperation visitminiJava__State(State it) {
		return new StateOperationImpl(it, this);
	}

	public FrameOperation visitminiJava__Frame(Frame it) {
		return new FrameOperationImpl(it, this);
	}

	public NullValueOperation visitminiJava__NullValue(NullValue it) {
		return new NullValueOperationImpl(it, this);
	}

	public NewCallOperation visitminiJava__NewCall(NewCall it) {
		return new NewCallOperationImpl(it, this);
	}

	public MethodCall2Operation visitminiJava__MethodCall2(MethodCall2 it) {
		return new MethodCall2OperationImpl(it, this);
	}

	public ObjectInstanceOperation visitminiJava__ObjectInstance(ObjectInstance it) {
		return new ObjectInstanceOperationImpl(it, this);
	}

	public ArrayInstanceOperation visitminiJava__ArrayInstance(ArrayInstance it) {
		return new ArrayInstanceOperationImpl(it, this);
	}

	public ObjectRefValueOperation visitminiJava__ObjectRefValue(ObjectRefValue it) {
		return new ObjectRefValueOperationImpl(it, this);
	}

	public ArrayRefValueOperation visitminiJava__ArrayRefValue(ArrayRefValue it) {
		return new ArrayRefValueOperationImpl(it, this);
	}

	public SymbolToSymbolBindingMapOperation visitminiJava__SymbolToSymbolBindingMap(
			SymbolToSymbolBindingMap it) {
		return new SymbolToSymbolBindingMapOperationImpl(it, this);
	}

	public ClazzToMethodMapOperation visitminiJava__ClazzToMethodMap(ClazzToMethodMap it) {
		return new ClazzToMethodMapOperationImpl(it, this);
	}

	public ModuloOperation visitminiJava__Modulo(Modulo it) {
		return new ModuloOperationImpl(it, this);
	}

	public SqrtOperation visitminiJava__Sqrt(Sqrt it) {
		return new SqrtOperationImpl(it, this);
	}
}
