package miniJava.revisitor;

public interface MiniJavaRevisitor<MiniJava__AndT extends MiniJava__ExpressionT, MiniJava__AndT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__AndT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__ArrayAccessT extends MiniJava__ExpressionT, MiniJava__ArrayAccessT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__ArrayAccessT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__ArrayInstanceT, MiniJava__ArrayLengthT extends MiniJava__ExpressionT, MiniJava__ArrayLengthT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__ArrayLengthT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__ArrayRefValueT extends MiniJava__ValueT, MiniJava__ArrayTypeRefT extends MiniJava__SingleTypeRefT, MiniJava__AssigneeT, MiniJava__AssignmentT extends MiniJava__StatementT, MiniJava__BlockT extends MiniJava__StatementT, MiniJava__BoolConstantT extends MiniJava__ExpressionT, MiniJava__BoolConstantT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__BoolConstantT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__BooleanTypeRefT extends MiniJava__SingleTypeRefT, MiniJava__BooleanValueT extends MiniJava__ValueT, MiniJava__CallT, MiniJava__ClassRefT extends MiniJava__SingleTypeRefT, MiniJava__ClazzT extends MiniJava__TypeDeclarationT, MiniJava__ContextT, MiniJava__DivisionT extends MiniJava__ExpressionT, MiniJava__DivisionT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__DivisionT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__EqualityT extends MiniJava__ExpressionT, MiniJava__EqualityT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__EqualityT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__ExpressionT, MiniJava__ExpressionT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__ExpressionT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__FieldT extends MiniJava__MemberT, MiniJava__FieldAccessT extends MiniJava__ExpressionT, MiniJava__FieldAccessT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__FieldAccessT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__FieldBindingT, MiniJava__ForStatementT extends MiniJava__StatementT, MiniJava__FrameT, MiniJava__IfStatementT extends MiniJava__StatementT, MiniJava__ImportT, MiniJava__InequalityT extends MiniJava__ExpressionT, MiniJava__InequalityT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__InequalityT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__InferiorT extends MiniJava__ExpressionT, MiniJava__InferiorT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__InferiorT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__InferiorOrEqualT extends MiniJava__ExpressionT, MiniJava__InferiorOrEqualT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__InferiorOrEqualT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__IntConstantT extends MiniJava__ExpressionT, MiniJava__IntConstantT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__IntConstantT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__IntegerTypeRefT extends MiniJava__SingleTypeRefT, MiniJava__IntegerValueT extends MiniJava__ValueT, MiniJava__InterfaceT extends MiniJava__TypeDeclarationT, MiniJava__LoadImageT extends MiniJava__ExpressionT, MiniJava__LoadImageT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__LoadImageT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__MemberT extends MiniJava__TypedDeclarationT, MiniJava__MethodT extends MiniJava__MemberT, MiniJava__MethodCallT extends MiniJava__ExpressionT, MiniJava__MethodCallT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__MethodCallT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__MethodCall2T extends MiniJava__CallT, MiniJava__MinusT extends MiniJava__ExpressionT, MiniJava__MinusT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__MinusT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__ModuloT extends MiniJava__ExpressionT, MiniJava__ModuloT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__ModuloT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__MultiplicationT extends MiniJava__ExpressionT, MiniJava__MultiplicationT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__MultiplicationT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__NamedElementT, MiniJava__NegT extends MiniJava__ExpressionT, MiniJava__NegT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__NegT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__NewArrayT extends MiniJava__ExpressionT, MiniJava__NewArrayT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__NewArrayT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__NewCallT extends MiniJava__CallT, MiniJava__NewObjectT extends MiniJava__ExpressionT, MiniJava__NewObjectT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__NewObjectT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__NotT extends MiniJava__ExpressionT, MiniJava__NotT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__NotT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__NullT extends MiniJava__ExpressionT, MiniJava__NullT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__NullT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__NullValueT extends MiniJava__ValueT, MiniJava__ObjectInstanceT, MiniJava__ObjectRefValueT extends MiniJava__ValueT, MiniJava__OrT extends MiniJava__ExpressionT, MiniJava__OrT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__OrT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__OutputStreamT, MiniJava__ParameterT extends MiniJava__SymbolT, MiniJava__PlusT extends MiniJava__ExpressionT, MiniJava__PlusT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__PlusT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__PrintStatementT extends MiniJava__StatementT, MiniJava__ProgramT, MiniJava__ReturnT extends MiniJava__StatementT, MiniJava__SingleTypeRefT extends MiniJava__TypeRefT, MiniJava__SqrtT extends MiniJava__ExpressionT, MiniJava__SqrtT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__SqrtT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__StateT, MiniJava__StatementT, MiniJava__StringConstantT extends MiniJava__ExpressionT, MiniJava__StringConstantT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__StringConstantT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__StringTypeRefT extends MiniJava__SingleTypeRefT, MiniJava__StringValueT extends MiniJava__ValueT, MiniJava__SuperT extends MiniJava__ExpressionT, MiniJava__SuperT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__SuperT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__SuperiorT extends MiniJava__ExpressionT, MiniJava__SuperiorT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__SuperiorT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__SuperiorOrEqualT extends MiniJava__ExpressionT, MiniJava__SuperiorOrEqualT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__SuperiorOrEqualT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__SymbolT extends MiniJava__TypedDeclarationT, MiniJava__SymbolBindingT, MiniJava__SymbolRefT extends MiniJava__ExpressionT, MiniJava__SymbolRefT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__SymbolRefT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__ThisT extends MiniJava__ExpressionT, MiniJava__ThisT_AS_MiniJava__StatementT extends MiniJava__StatementT, MiniJava__ThisT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__TypeDeclarationT extends MiniJava__NamedElementT, MiniJava__TypeRefT, MiniJava__TypedDeclarationT extends MiniJava__NamedElementT, MiniJava__ValueT, MiniJava__VariableDeclarationT, MiniJava__VariableDeclarationT_AS_MiniJava__SymbolT extends MiniJava__SymbolT, MiniJava__VariableDeclarationT_AS_MiniJava__AssigneeT extends MiniJava__AssigneeT, MiniJava__VoidTypeRefT extends MiniJava__SingleTypeRefT, MiniJava__WhileStatementT extends MiniJava__StatementT, MiniJava__WriteImageT extends MiniJava__StatementT> {
	MiniJava__AndT miniJava__And(final miniJava.And it);
	MiniJava__AndT_AS_MiniJava__StatementT miniJava__And__AS__miniJava__Statement(final miniJava.And it);
	MiniJava__AndT_AS_MiniJava__AssigneeT miniJava__And__AS__miniJava__Assignee(final miniJava.And it);
	MiniJava__ArrayAccessT miniJava__ArrayAccess(final miniJava.ArrayAccess it);
	MiniJava__ArrayAccessT_AS_MiniJava__StatementT miniJava__ArrayAccess__AS__miniJava__Statement(final miniJava.ArrayAccess it);
	MiniJava__ArrayAccessT_AS_MiniJava__AssigneeT miniJava__ArrayAccess__AS__miniJava__Assignee(final miniJava.ArrayAccess it);
	MiniJava__ArrayInstanceT miniJava__ArrayInstance(final miniJava.ArrayInstance it);
	MiniJava__ArrayLengthT miniJava__ArrayLength(final miniJava.ArrayLength it);
	MiniJava__ArrayLengthT_AS_MiniJava__StatementT miniJava__ArrayLength__AS__miniJava__Statement(final miniJava.ArrayLength it);
	MiniJava__ArrayLengthT_AS_MiniJava__AssigneeT miniJava__ArrayLength__AS__miniJava__Assignee(final miniJava.ArrayLength it);
	MiniJava__ArrayRefValueT miniJava__ArrayRefValue(final miniJava.ArrayRefValue it);
	MiniJava__ArrayTypeRefT miniJava__ArrayTypeRef(final miniJava.ArrayTypeRef it);
	MiniJava__AssigneeT miniJava__Assignee(final miniJava.Assignee it);
	MiniJava__AssignmentT miniJava__Assignment(final miniJava.Assignment it);
	MiniJava__BlockT miniJava__Block(final miniJava.Block it);
	MiniJava__BoolConstantT miniJava__BoolConstant(final miniJava.BoolConstant it);
	MiniJava__BoolConstantT_AS_MiniJava__StatementT miniJava__BoolConstant__AS__miniJava__Statement(final miniJava.BoolConstant it);
	MiniJava__BoolConstantT_AS_MiniJava__AssigneeT miniJava__BoolConstant__AS__miniJava__Assignee(final miniJava.BoolConstant it);
	MiniJava__BooleanTypeRefT miniJava__BooleanTypeRef(final miniJava.BooleanTypeRef it);
	MiniJava__BooleanValueT miniJava__BooleanValue(final miniJava.BooleanValue it);
	MiniJava__ClassRefT miniJava__ClassRef(final miniJava.ClassRef it);
	MiniJava__ClazzT miniJava__Clazz(final miniJava.Clazz it);
	MiniJava__ContextT miniJava__Context(final miniJava.Context it);
	MiniJava__DivisionT miniJava__Division(final miniJava.Division it);
	MiniJava__DivisionT_AS_MiniJava__StatementT miniJava__Division__AS__miniJava__Statement(final miniJava.Division it);
	MiniJava__DivisionT_AS_MiniJava__AssigneeT miniJava__Division__AS__miniJava__Assignee(final miniJava.Division it);
	MiniJava__EqualityT miniJava__Equality(final miniJava.Equality it);
	MiniJava__EqualityT_AS_MiniJava__StatementT miniJava__Equality__AS__miniJava__Statement(final miniJava.Equality it);
	MiniJava__EqualityT_AS_MiniJava__AssigneeT miniJava__Equality__AS__miniJava__Assignee(final miniJava.Equality it);
	MiniJava__ExpressionT miniJava__Expression(final miniJava.Expression it);
	MiniJava__ExpressionT_AS_MiniJava__StatementT miniJava__Expression__AS__miniJava__Statement(final miniJava.Expression it);
	MiniJava__ExpressionT_AS_MiniJava__AssigneeT miniJava__Expression__AS__miniJava__Assignee(final miniJava.Expression it);
	MiniJava__FieldT miniJava__Field(final miniJava.Field it);
	MiniJava__FieldAccessT miniJava__FieldAccess(final miniJava.FieldAccess it);
	MiniJava__FieldAccessT_AS_MiniJava__StatementT miniJava__FieldAccess__AS__miniJava__Statement(final miniJava.FieldAccess it);
	MiniJava__FieldAccessT_AS_MiniJava__AssigneeT miniJava__FieldAccess__AS__miniJava__Assignee(final miniJava.FieldAccess it);
	MiniJava__FieldBindingT miniJava__FieldBinding(final miniJava.FieldBinding it);
	MiniJava__ForStatementT miniJava__ForStatement(final miniJava.ForStatement it);
	MiniJava__FrameT miniJava__Frame(final miniJava.Frame it);
	MiniJava__IfStatementT miniJava__IfStatement(final miniJava.IfStatement it);
	MiniJava__ImportT miniJava__Import(final miniJava.Import it);
	MiniJava__InequalityT miniJava__Inequality(final miniJava.Inequality it);
	MiniJava__InequalityT_AS_MiniJava__StatementT miniJava__Inequality__AS__miniJava__Statement(final miniJava.Inequality it);
	MiniJava__InequalityT_AS_MiniJava__AssigneeT miniJava__Inequality__AS__miniJava__Assignee(final miniJava.Inequality it);
	MiniJava__InferiorT miniJava__Inferior(final miniJava.Inferior it);
	MiniJava__InferiorT_AS_MiniJava__StatementT miniJava__Inferior__AS__miniJava__Statement(final miniJava.Inferior it);
	MiniJava__InferiorT_AS_MiniJava__AssigneeT miniJava__Inferior__AS__miniJava__Assignee(final miniJava.Inferior it);
	MiniJava__InferiorOrEqualT miniJava__InferiorOrEqual(final miniJava.InferiorOrEqual it);
	MiniJava__InferiorOrEqualT_AS_MiniJava__StatementT miniJava__InferiorOrEqual__AS__miniJava__Statement(final miniJava.InferiorOrEqual it);
	MiniJava__InferiorOrEqualT_AS_MiniJava__AssigneeT miniJava__InferiorOrEqual__AS__miniJava__Assignee(final miniJava.InferiorOrEqual it);
	MiniJava__IntConstantT miniJava__IntConstant(final miniJava.IntConstant it);
	MiniJava__IntConstantT_AS_MiniJava__StatementT miniJava__IntConstant__AS__miniJava__Statement(final miniJava.IntConstant it);
	MiniJava__IntConstantT_AS_MiniJava__AssigneeT miniJava__IntConstant__AS__miniJava__Assignee(final miniJava.IntConstant it);
	MiniJava__IntegerTypeRefT miniJava__IntegerTypeRef(final miniJava.IntegerTypeRef it);
	MiniJava__IntegerValueT miniJava__IntegerValue(final miniJava.IntegerValue it);
	MiniJava__InterfaceT miniJava__Interface(final miniJava.Interface it);
	MiniJava__LoadImageT miniJava__LoadImage(final miniJava.LoadImage it);
	MiniJava__LoadImageT_AS_MiniJava__StatementT miniJava__LoadImage__AS__miniJava__Statement(final miniJava.LoadImage it);
	MiniJava__LoadImageT_AS_MiniJava__AssigneeT miniJava__LoadImage__AS__miniJava__Assignee(final miniJava.LoadImage it);
	MiniJava__MemberT miniJava__Member(final miniJava.Member it);
	MiniJava__MethodT miniJava__Method(final miniJava.Method it);
	MiniJava__MethodCallT miniJava__MethodCall(final miniJava.MethodCall it);
	MiniJava__MethodCallT_AS_MiniJava__StatementT miniJava__MethodCall__AS__miniJava__Statement(final miniJava.MethodCall it);
	MiniJava__MethodCallT_AS_MiniJava__AssigneeT miniJava__MethodCall__AS__miniJava__Assignee(final miniJava.MethodCall it);
	MiniJava__MethodCall2T miniJava__MethodCall2(final miniJava.MethodCall2 it);
	MiniJava__MinusT miniJava__Minus(final miniJava.Minus it);
	MiniJava__MinusT_AS_MiniJava__StatementT miniJava__Minus__AS__miniJava__Statement(final miniJava.Minus it);
	MiniJava__MinusT_AS_MiniJava__AssigneeT miniJava__Minus__AS__miniJava__Assignee(final miniJava.Minus it);
	MiniJava__ModuloT miniJava__Modulo(final miniJava.Modulo it);
	MiniJava__ModuloT_AS_MiniJava__StatementT miniJava__Modulo__AS__miniJava__Statement(final miniJava.Modulo it);
	MiniJava__ModuloT_AS_MiniJava__AssigneeT miniJava__Modulo__AS__miniJava__Assignee(final miniJava.Modulo it);
	MiniJava__MultiplicationT miniJava__Multiplication(final miniJava.Multiplication it);
	MiniJava__MultiplicationT_AS_MiniJava__StatementT miniJava__Multiplication__AS__miniJava__Statement(final miniJava.Multiplication it);
	MiniJava__MultiplicationT_AS_MiniJava__AssigneeT miniJava__Multiplication__AS__miniJava__Assignee(final miniJava.Multiplication it);
	MiniJava__NamedElementT miniJava__NamedElement(final miniJava.NamedElement it);
	MiniJava__NegT miniJava__Neg(final miniJava.Neg it);
	MiniJava__NegT_AS_MiniJava__StatementT miniJava__Neg__AS__miniJava__Statement(final miniJava.Neg it);
	MiniJava__NegT_AS_MiniJava__AssigneeT miniJava__Neg__AS__miniJava__Assignee(final miniJava.Neg it);
	MiniJava__NewArrayT miniJava__NewArray(final miniJava.NewArray it);
	MiniJava__NewArrayT_AS_MiniJava__StatementT miniJava__NewArray__AS__miniJava__Statement(final miniJava.NewArray it);
	MiniJava__NewArrayT_AS_MiniJava__AssigneeT miniJava__NewArray__AS__miniJava__Assignee(final miniJava.NewArray it);
	MiniJava__NewCallT miniJava__NewCall(final miniJava.NewCall it);
	MiniJava__NewObjectT miniJava__NewObject(final miniJava.NewObject it);
	MiniJava__NewObjectT_AS_MiniJava__StatementT miniJava__NewObject__AS__miniJava__Statement(final miniJava.NewObject it);
	MiniJava__NewObjectT_AS_MiniJava__AssigneeT miniJava__NewObject__AS__miniJava__Assignee(final miniJava.NewObject it);
	MiniJava__NotT miniJava__Not(final miniJava.Not it);
	MiniJava__NotT_AS_MiniJava__StatementT miniJava__Not__AS__miniJava__Statement(final miniJava.Not it);
	MiniJava__NotT_AS_MiniJava__AssigneeT miniJava__Not__AS__miniJava__Assignee(final miniJava.Not it);
	MiniJava__NullT miniJava__Null(final miniJava.Null it);
	MiniJava__NullT_AS_MiniJava__StatementT miniJava__Null__AS__miniJava__Statement(final miniJava.Null it);
	MiniJava__NullT_AS_MiniJava__AssigneeT miniJava__Null__AS__miniJava__Assignee(final miniJava.Null it);
	MiniJava__NullValueT miniJava__NullValue(final miniJava.NullValue it);
	MiniJava__ObjectInstanceT miniJava__ObjectInstance(final miniJava.ObjectInstance it);
	MiniJava__ObjectRefValueT miniJava__ObjectRefValue(final miniJava.ObjectRefValue it);
	MiniJava__OrT miniJava__Or(final miniJava.Or it);
	MiniJava__OrT_AS_MiniJava__StatementT miniJava__Or__AS__miniJava__Statement(final miniJava.Or it);
	MiniJava__OrT_AS_MiniJava__AssigneeT miniJava__Or__AS__miniJava__Assignee(final miniJava.Or it);
	MiniJava__OutputStreamT miniJava__OutputStream(final miniJava.OutputStream it);
	MiniJava__ParameterT miniJava__Parameter(final miniJava.Parameter it);
	MiniJava__PlusT miniJava__Plus(final miniJava.Plus it);
	MiniJava__PlusT_AS_MiniJava__StatementT miniJava__Plus__AS__miniJava__Statement(final miniJava.Plus it);
	MiniJava__PlusT_AS_MiniJava__AssigneeT miniJava__Plus__AS__miniJava__Assignee(final miniJava.Plus it);
	MiniJava__PrintStatementT miniJava__PrintStatement(final miniJava.PrintStatement it);
	MiniJava__ProgramT miniJava__Program(final miniJava.Program it);
	MiniJava__ReturnT miniJava__Return(final miniJava.Return it);
	MiniJava__SingleTypeRefT miniJava__SingleTypeRef(final miniJava.SingleTypeRef it);
	MiniJava__SqrtT miniJava__Sqrt(final miniJava.Sqrt it);
	MiniJava__SqrtT_AS_MiniJava__StatementT miniJava__Sqrt__AS__miniJava__Statement(final miniJava.Sqrt it);
	MiniJava__SqrtT_AS_MiniJava__AssigneeT miniJava__Sqrt__AS__miniJava__Assignee(final miniJava.Sqrt it);
	MiniJava__StateT miniJava__State(final miniJava.State it);
	MiniJava__StatementT miniJava__Statement(final miniJava.Statement it);
	MiniJava__StringConstantT miniJava__StringConstant(final miniJava.StringConstant it);
	MiniJava__StringConstantT_AS_MiniJava__StatementT miniJava__StringConstant__AS__miniJava__Statement(final miniJava.StringConstant it);
	MiniJava__StringConstantT_AS_MiniJava__AssigneeT miniJava__StringConstant__AS__miniJava__Assignee(final miniJava.StringConstant it);
	MiniJava__StringTypeRefT miniJava__StringTypeRef(final miniJava.StringTypeRef it);
	MiniJava__StringValueT miniJava__StringValue(final miniJava.StringValue it);
	MiniJava__SuperT miniJava__Super(final miniJava.Super it);
	MiniJava__SuperT_AS_MiniJava__StatementT miniJava__Super__AS__miniJava__Statement(final miniJava.Super it);
	MiniJava__SuperT_AS_MiniJava__AssigneeT miniJava__Super__AS__miniJava__Assignee(final miniJava.Super it);
	MiniJava__SuperiorT miniJava__Superior(final miniJava.Superior it);
	MiniJava__SuperiorT_AS_MiniJava__StatementT miniJava__Superior__AS__miniJava__Statement(final miniJava.Superior it);
	MiniJava__SuperiorT_AS_MiniJava__AssigneeT miniJava__Superior__AS__miniJava__Assignee(final miniJava.Superior it);
	MiniJava__SuperiorOrEqualT miniJava__SuperiorOrEqual(final miniJava.SuperiorOrEqual it);
	MiniJava__SuperiorOrEqualT_AS_MiniJava__StatementT miniJava__SuperiorOrEqual__AS__miniJava__Statement(final miniJava.SuperiorOrEqual it);
	MiniJava__SuperiorOrEqualT_AS_MiniJava__AssigneeT miniJava__SuperiorOrEqual__AS__miniJava__Assignee(final miniJava.SuperiorOrEqual it);
	MiniJava__SymbolT miniJava__Symbol(final miniJava.Symbol it);
	MiniJava__SymbolBindingT miniJava__SymbolBinding(final miniJava.SymbolBinding it);
	MiniJava__SymbolRefT miniJava__SymbolRef(final miniJava.SymbolRef it);
	MiniJava__SymbolRefT_AS_MiniJava__StatementT miniJava__SymbolRef__AS__miniJava__Statement(final miniJava.SymbolRef it);
	MiniJava__SymbolRefT_AS_MiniJava__AssigneeT miniJava__SymbolRef__AS__miniJava__Assignee(final miniJava.SymbolRef it);
	MiniJava__ThisT miniJava__This(final miniJava.This it);
	MiniJava__ThisT_AS_MiniJava__StatementT miniJava__This__AS__miniJava__Statement(final miniJava.This it);
	MiniJava__ThisT_AS_MiniJava__AssigneeT miniJava__This__AS__miniJava__Assignee(final miniJava.This it);
	MiniJava__TypeDeclarationT miniJava__TypeDeclaration(final miniJava.TypeDeclaration it);
	MiniJava__TypeRefT miniJava__TypeRef(final miniJava.TypeRef it);
	MiniJava__TypedDeclarationT miniJava__TypedDeclaration(final miniJava.TypedDeclaration it);
	MiniJava__ValueT miniJava__Value(final miniJava.Value it);
	MiniJava__VariableDeclarationT miniJava__VariableDeclaration(final miniJava.VariableDeclaration it);
	MiniJava__VariableDeclarationT_AS_MiniJava__SymbolT miniJava__VariableDeclaration__AS__miniJava__Symbol(final miniJava.VariableDeclaration it);
	MiniJava__VariableDeclarationT_AS_MiniJava__AssigneeT miniJava__VariableDeclaration__AS__miniJava__Assignee(final miniJava.VariableDeclaration it);
	MiniJava__VoidTypeRefT miniJava__VoidTypeRef(final miniJava.VoidTypeRef it);
	MiniJava__WhileStatementT miniJava__WhileStatement(final miniJava.WhileStatement it);
	MiniJava__WriteImageT miniJava__WriteImage(final miniJava.WriteImage it);

	default MiniJava__AndT $(final miniJava.And it) {
		return miniJava__And(it);
	}
	default MiniJava__ArrayAccessT $(final miniJava.ArrayAccess it) {
		return miniJava__ArrayAccess(it);
	}
	default MiniJava__ArrayInstanceT $(final miniJava.ArrayInstance it) {
		return miniJava__ArrayInstance(it);
	}
	default MiniJava__ArrayLengthT $(final miniJava.ArrayLength it) {
		return miniJava__ArrayLength(it);
	}
	default MiniJava__ArrayRefValueT $(final miniJava.ArrayRefValue it) {
		return miniJava__ArrayRefValue(it);
	}
	default MiniJava__ArrayTypeRefT $(final miniJava.ArrayTypeRef it) {
		return miniJava__ArrayTypeRef(it);
	}
	default MiniJava__AssigneeT $(final miniJava.Assignee it) {
		if (it.getClass() == miniJava.impl.AndImpl.class)
			return miniJava__And__AS__miniJava__Assignee((miniJava.And) it);
		if (it.getClass() == miniJava.impl.ArrayAccessImpl.class)
			return miniJava__ArrayAccess__AS__miniJava__Assignee((miniJava.ArrayAccess) it);
		if (it.getClass() == miniJava.impl.ArrayLengthImpl.class)
			return miniJava__ArrayLength__AS__miniJava__Assignee((miniJava.ArrayLength) it);
		if (it.getClass() == miniJava.impl.BoolConstantImpl.class)
			return miniJava__BoolConstant__AS__miniJava__Assignee((miniJava.BoolConstant) it);
		if (it.getClass() == miniJava.impl.DivisionImpl.class)
			return miniJava__Division__AS__miniJava__Assignee((miniJava.Division) it);
		if (it.getClass() == miniJava.impl.EqualityImpl.class)
			return miniJava__Equality__AS__miniJava__Assignee((miniJava.Equality) it);
		if (it.getClass() == miniJava.impl.ExpressionImpl.class)
			return miniJava__Expression__AS__miniJava__Assignee((miniJava.Expression) it);
		if (it.getClass() == miniJava.impl.FieldAccessImpl.class)
			return miniJava__FieldAccess__AS__miniJava__Assignee((miniJava.FieldAccess) it);
		if (it.getClass() == miniJava.impl.InequalityImpl.class)
			return miniJava__Inequality__AS__miniJava__Assignee((miniJava.Inequality) it);
		if (it.getClass() == miniJava.impl.InferiorImpl.class)
			return miniJava__Inferior__AS__miniJava__Assignee((miniJava.Inferior) it);
		if (it.getClass() == miniJava.impl.InferiorOrEqualImpl.class)
			return miniJava__InferiorOrEqual__AS__miniJava__Assignee((miniJava.InferiorOrEqual) it);
		if (it.getClass() == miniJava.impl.IntConstantImpl.class)
			return miniJava__IntConstant__AS__miniJava__Assignee((miniJava.IntConstant) it);
		if (it.getClass() == miniJava.impl.LoadImageImpl.class)
			return miniJava__LoadImage__AS__miniJava__Assignee((miniJava.LoadImage) it);
		if (it.getClass() == miniJava.impl.MethodCallImpl.class)
			return miniJava__MethodCall__AS__miniJava__Assignee((miniJava.MethodCall) it);
		if (it.getClass() == miniJava.impl.MinusImpl.class)
			return miniJava__Minus__AS__miniJava__Assignee((miniJava.Minus) it);
		if (it.getClass() == miniJava.impl.ModuloImpl.class)
			return miniJava__Modulo__AS__miniJava__Assignee((miniJava.Modulo) it);
		if (it.getClass() == miniJava.impl.MultiplicationImpl.class)
			return miniJava__Multiplication__AS__miniJava__Assignee((miniJava.Multiplication) it);
		if (it.getClass() == miniJava.impl.NegImpl.class)
			return miniJava__Neg__AS__miniJava__Assignee((miniJava.Neg) it);
		if (it.getClass() == miniJava.impl.NewArrayImpl.class)
			return miniJava__NewArray__AS__miniJava__Assignee((miniJava.NewArray) it);
		if (it.getClass() == miniJava.impl.NewObjectImpl.class)
			return miniJava__NewObject__AS__miniJava__Assignee((miniJava.NewObject) it);
		if (it.getClass() == miniJava.impl.NotImpl.class)
			return miniJava__Not__AS__miniJava__Assignee((miniJava.Not) it);
		if (it.getClass() == miniJava.impl.NullImpl.class)
			return miniJava__Null__AS__miniJava__Assignee((miniJava.Null) it);
		if (it.getClass() == miniJava.impl.OrImpl.class)
			return miniJava__Or__AS__miniJava__Assignee((miniJava.Or) it);
		if (it.getClass() == miniJava.impl.PlusImpl.class)
			return miniJava__Plus__AS__miniJava__Assignee((miniJava.Plus) it);
		if (it.getClass() == miniJava.impl.SqrtImpl.class)
			return miniJava__Sqrt__AS__miniJava__Assignee((miniJava.Sqrt) it);
		if (it.getClass() == miniJava.impl.StringConstantImpl.class)
			return miniJava__StringConstant__AS__miniJava__Assignee((miniJava.StringConstant) it);
		if (it.getClass() == miniJava.impl.SuperImpl.class)
			return miniJava__Super__AS__miniJava__Assignee((miniJava.Super) it);
		if (it.getClass() == miniJava.impl.SuperiorImpl.class)
			return miniJava__Superior__AS__miniJava__Assignee((miniJava.Superior) it);
		if (it.getClass() == miniJava.impl.SuperiorOrEqualImpl.class)
			return miniJava__SuperiorOrEqual__AS__miniJava__Assignee((miniJava.SuperiorOrEqual) it);
		if (it.getClass() == miniJava.impl.SymbolRefImpl.class)
			return miniJava__SymbolRef__AS__miniJava__Assignee((miniJava.SymbolRef) it);
		if (it.getClass() == miniJava.impl.ThisImpl.class)
			return miniJava__This__AS__miniJava__Assignee((miniJava.This) it);
		if (it.getClass() == miniJava.impl.VariableDeclarationImpl.class)
			return miniJava__VariableDeclaration__AS__miniJava__Assignee((miniJava.VariableDeclaration) it);
		return miniJava__Assignee(it);
	}
	default MiniJava__AssignmentT $(final miniJava.Assignment it) {
		return miniJava__Assignment(it);
	}
	default MiniJava__BlockT $(final miniJava.Block it) {
		return miniJava__Block(it);
	}
	default MiniJava__BoolConstantT $(final miniJava.BoolConstant it) {
		return miniJava__BoolConstant(it);
	}
	default MiniJava__BooleanTypeRefT $(final miniJava.BooleanTypeRef it) {
		return miniJava__BooleanTypeRef(it);
	}
	default MiniJava__BooleanValueT $(final miniJava.BooleanValue it) {
		return miniJava__BooleanValue(it);
	}
	default MiniJava__CallT $(final miniJava.Call it) {
		if (it.getClass() == miniJava.impl.MethodCall2Impl.class)
			return miniJava__MethodCall2((miniJava.MethodCall2) it);
		if (it.getClass() == miniJava.impl.NewCallImpl.class)
			return miniJava__NewCall((miniJava.NewCall) it);
		return null;
	}
	default MiniJava__ClassRefT $(final miniJava.ClassRef it) {
		return miniJava__ClassRef(it);
	}
	default MiniJava__ClazzT $(final miniJava.Clazz it) {
		return miniJava__Clazz(it);
	}
	default MiniJava__ContextT $(final miniJava.Context it) {
		return miniJava__Context(it);
	}
	default MiniJava__DivisionT $(final miniJava.Division it) {
		return miniJava__Division(it);
	}
	default MiniJava__EqualityT $(final miniJava.Equality it) {
		return miniJava__Equality(it);
	}
	default MiniJava__ExpressionT $(final miniJava.Expression it) {
		if (it.getClass() == miniJava.impl.AndImpl.class)
			return miniJava__And((miniJava.And) it);
		if (it.getClass() == miniJava.impl.ArrayAccessImpl.class)
			return miniJava__ArrayAccess((miniJava.ArrayAccess) it);
		if (it.getClass() == miniJava.impl.ArrayLengthImpl.class)
			return miniJava__ArrayLength((miniJava.ArrayLength) it);
		if (it.getClass() == miniJava.impl.BoolConstantImpl.class)
			return miniJava__BoolConstant((miniJava.BoolConstant) it);
		if (it.getClass() == miniJava.impl.DivisionImpl.class)
			return miniJava__Division((miniJava.Division) it);
		if (it.getClass() == miniJava.impl.EqualityImpl.class)
			return miniJava__Equality((miniJava.Equality) it);
		if (it.getClass() == miniJava.impl.FieldAccessImpl.class)
			return miniJava__FieldAccess((miniJava.FieldAccess) it);
		if (it.getClass() == miniJava.impl.InequalityImpl.class)
			return miniJava__Inequality((miniJava.Inequality) it);
		if (it.getClass() == miniJava.impl.InferiorImpl.class)
			return miniJava__Inferior((miniJava.Inferior) it);
		if (it.getClass() == miniJava.impl.InferiorOrEqualImpl.class)
			return miniJava__InferiorOrEqual((miniJava.InferiorOrEqual) it);
		if (it.getClass() == miniJava.impl.IntConstantImpl.class)
			return miniJava__IntConstant((miniJava.IntConstant) it);
		if (it.getClass() == miniJava.impl.LoadImageImpl.class)
			return miniJava__LoadImage((miniJava.LoadImage) it);
		if (it.getClass() == miniJava.impl.MethodCallImpl.class)
			return miniJava__MethodCall((miniJava.MethodCall) it);
		if (it.getClass() == miniJava.impl.MinusImpl.class)
			return miniJava__Minus((miniJava.Minus) it);
		if (it.getClass() == miniJava.impl.ModuloImpl.class)
			return miniJava__Modulo((miniJava.Modulo) it);
		if (it.getClass() == miniJava.impl.MultiplicationImpl.class)
			return miniJava__Multiplication((miniJava.Multiplication) it);
		if (it.getClass() == miniJava.impl.NegImpl.class)
			return miniJava__Neg((miniJava.Neg) it);
		if (it.getClass() == miniJava.impl.NewArrayImpl.class)
			return miniJava__NewArray((miniJava.NewArray) it);
		if (it.getClass() == miniJava.impl.NewObjectImpl.class)
			return miniJava__NewObject((miniJava.NewObject) it);
		if (it.getClass() == miniJava.impl.NotImpl.class)
			return miniJava__Not((miniJava.Not) it);
		if (it.getClass() == miniJava.impl.NullImpl.class)
			return miniJava__Null((miniJava.Null) it);
		if (it.getClass() == miniJava.impl.OrImpl.class)
			return miniJava__Or((miniJava.Or) it);
		if (it.getClass() == miniJava.impl.PlusImpl.class)
			return miniJava__Plus((miniJava.Plus) it);
		if (it.getClass() == miniJava.impl.SqrtImpl.class)
			return miniJava__Sqrt((miniJava.Sqrt) it);
		if (it.getClass() == miniJava.impl.StringConstantImpl.class)
			return miniJava__StringConstant((miniJava.StringConstant) it);
		if (it.getClass() == miniJava.impl.SuperImpl.class)
			return miniJava__Super((miniJava.Super) it);
		if (it.getClass() == miniJava.impl.SuperiorImpl.class)
			return miniJava__Superior((miniJava.Superior) it);
		if (it.getClass() == miniJava.impl.SuperiorOrEqualImpl.class)
			return miniJava__SuperiorOrEqual((miniJava.SuperiorOrEqual) it);
		if (it.getClass() == miniJava.impl.SymbolRefImpl.class)
			return miniJava__SymbolRef((miniJava.SymbolRef) it);
		if (it.getClass() == miniJava.impl.ThisImpl.class)
			return miniJava__This((miniJava.This) it);
		return miniJava__Expression(it);
	}
	default MiniJava__FieldT $(final miniJava.Field it) {
		return miniJava__Field(it);
	}
	default MiniJava__FieldAccessT $(final miniJava.FieldAccess it) {
		return miniJava__FieldAccess(it);
	}
	default MiniJava__FieldBindingT $(final miniJava.FieldBinding it) {
		return miniJava__FieldBinding(it);
	}
	default MiniJava__ForStatementT $(final miniJava.ForStatement it) {
		return miniJava__ForStatement(it);
	}
	default MiniJava__FrameT $(final miniJava.Frame it) {
		return miniJava__Frame(it);
	}
	default MiniJava__IfStatementT $(final miniJava.IfStatement it) {
		return miniJava__IfStatement(it);
	}
	default MiniJava__ImportT $(final miniJava.Import it) {
		return miniJava__Import(it);
	}
	default MiniJava__InequalityT $(final miniJava.Inequality it) {
		return miniJava__Inequality(it);
	}
	default MiniJava__InferiorT $(final miniJava.Inferior it) {
		return miniJava__Inferior(it);
	}
	default MiniJava__InferiorOrEqualT $(final miniJava.InferiorOrEqual it) {
		return miniJava__InferiorOrEqual(it);
	}
	default MiniJava__IntConstantT $(final miniJava.IntConstant it) {
		return miniJava__IntConstant(it);
	}
	default MiniJava__IntegerTypeRefT $(final miniJava.IntegerTypeRef it) {
		return miniJava__IntegerTypeRef(it);
	}
	default MiniJava__IntegerValueT $(final miniJava.IntegerValue it) {
		return miniJava__IntegerValue(it);
	}
	default MiniJava__InterfaceT $(final miniJava.Interface it) {
		return miniJava__Interface(it);
	}
	default MiniJava__LoadImageT $(final miniJava.LoadImage it) {
		return miniJava__LoadImage(it);
	}
	default MiniJava__MemberT $(final miniJava.Member it) {
		if (it.getClass() == miniJava.impl.FieldImpl.class)
			return miniJava__Field((miniJava.Field) it);
		if (it.getClass() == miniJava.impl.MethodImpl.class)
			return miniJava__Method((miniJava.Method) it);
		return miniJava__Member(it);
	}
	default MiniJava__MethodT $(final miniJava.Method it) {
		return miniJava__Method(it);
	}
	default MiniJava__MethodCallT $(final miniJava.MethodCall it) {
		return miniJava__MethodCall(it);
	}
	default MiniJava__MethodCall2T $(final miniJava.MethodCall2 it) {
		return miniJava__MethodCall2(it);
	}
	default MiniJava__MinusT $(final miniJava.Minus it) {
		return miniJava__Minus(it);
	}
	default MiniJava__ModuloT $(final miniJava.Modulo it) {
		return miniJava__Modulo(it);
	}
	default MiniJava__MultiplicationT $(final miniJava.Multiplication it) {
		return miniJava__Multiplication(it);
	}
	default MiniJava__NamedElementT $(final miniJava.NamedElement it) {
		if (it.getClass() == miniJava.impl.ClazzImpl.class)
			return miniJava__Clazz((miniJava.Clazz) it);
		if (it.getClass() == miniJava.impl.FieldImpl.class)
			return miniJava__Field((miniJava.Field) it);
		if (it.getClass() == miniJava.impl.InterfaceImpl.class)
			return miniJava__Interface((miniJava.Interface) it);
		if (it.getClass() == miniJava.impl.MemberImpl.class)
			return miniJava__Member((miniJava.Member) it);
		if (it.getClass() == miniJava.impl.MethodImpl.class)
			return miniJava__Method((miniJava.Method) it);
		if (it.getClass() == miniJava.impl.ParameterImpl.class)
			return miniJava__Parameter((miniJava.Parameter) it);
		if (it.getClass() == miniJava.impl.SymbolImpl.class)
			return miniJava__Symbol((miniJava.Symbol) it);
		if (it.getClass() == miniJava.impl.TypeDeclarationImpl.class)
			return miniJava__TypeDeclaration((miniJava.TypeDeclaration) it);
		if (it.getClass() == miniJava.impl.TypedDeclarationImpl.class)
			return miniJava__TypedDeclaration((miniJava.TypedDeclaration) it);
		if (it.getClass() == miniJava.impl.VariableDeclarationImpl.class)
			return miniJava__VariableDeclaration__AS__miniJava__Symbol((miniJava.VariableDeclaration) it);
		return miniJava__NamedElement(it);
	}
	default MiniJava__NegT $(final miniJava.Neg it) {
		return miniJava__Neg(it);
	}
	default MiniJava__NewArrayT $(final miniJava.NewArray it) {
		return miniJava__NewArray(it);
	}
	default MiniJava__NewCallT $(final miniJava.NewCall it) {
		return miniJava__NewCall(it);
	}
	default MiniJava__NewObjectT $(final miniJava.NewObject it) {
		return miniJava__NewObject(it);
	}
	default MiniJava__NotT $(final miniJava.Not it) {
		return miniJava__Not(it);
	}
	default MiniJava__NullT $(final miniJava.Null it) {
		return miniJava__Null(it);
	}
	default MiniJava__NullValueT $(final miniJava.NullValue it) {
		return miniJava__NullValue(it);
	}
	default MiniJava__ObjectInstanceT $(final miniJava.ObjectInstance it) {
		return miniJava__ObjectInstance(it);
	}
	default MiniJava__ObjectRefValueT $(final miniJava.ObjectRefValue it) {
		return miniJava__ObjectRefValue(it);
	}
	default MiniJava__OrT $(final miniJava.Or it) {
		return miniJava__Or(it);
	}
	default MiniJava__OutputStreamT $(final miniJava.OutputStream it) {
		return miniJava__OutputStream(it);
	}
	default MiniJava__ParameterT $(final miniJava.Parameter it) {
		return miniJava__Parameter(it);
	}
	default MiniJava__PlusT $(final miniJava.Plus it) {
		return miniJava__Plus(it);
	}
	default MiniJava__PrintStatementT $(final miniJava.PrintStatement it) {
		return miniJava__PrintStatement(it);
	}
	default MiniJava__ProgramT $(final miniJava.Program it) {
		return miniJava__Program(it);
	}
	default MiniJava__ReturnT $(final miniJava.Return it) {
		return miniJava__Return(it);
	}
	default MiniJava__SingleTypeRefT $(final miniJava.SingleTypeRef it) {
		if (it.getClass() == miniJava.impl.ArrayTypeRefImpl.class)
			return miniJava__ArrayTypeRef((miniJava.ArrayTypeRef) it);
		if (it.getClass() == miniJava.impl.BooleanTypeRefImpl.class)
			return miniJava__BooleanTypeRef((miniJava.BooleanTypeRef) it);
		if (it.getClass() == miniJava.impl.ClassRefImpl.class)
			return miniJava__ClassRef((miniJava.ClassRef) it);
		if (it.getClass() == miniJava.impl.IntegerTypeRefImpl.class)
			return miniJava__IntegerTypeRef((miniJava.IntegerTypeRef) it);
		if (it.getClass() == miniJava.impl.StringTypeRefImpl.class)
			return miniJava__StringTypeRef((miniJava.StringTypeRef) it);
		if (it.getClass() == miniJava.impl.VoidTypeRefImpl.class)
			return miniJava__VoidTypeRef((miniJava.VoidTypeRef) it);
		return miniJava__SingleTypeRef(it);
	}
	default MiniJava__SqrtT $(final miniJava.Sqrt it) {
		return miniJava__Sqrt(it);
	}
	default MiniJava__StateT $(final miniJava.State it) {
		return miniJava__State(it);
	}
	default MiniJava__StatementT $(final miniJava.Statement it) {
		if (it.getClass() == miniJava.impl.AndImpl.class)
			return miniJava__And__AS__miniJava__Statement((miniJava.And) it);
		if (it.getClass() == miniJava.impl.ArrayAccessImpl.class)
			return miniJava__ArrayAccess__AS__miniJava__Statement((miniJava.ArrayAccess) it);
		if (it.getClass() == miniJava.impl.ArrayLengthImpl.class)
			return miniJava__ArrayLength__AS__miniJava__Statement((miniJava.ArrayLength) it);
		if (it.getClass() == miniJava.impl.AssignmentImpl.class)
			return miniJava__Assignment((miniJava.Assignment) it);
		if (it.getClass() == miniJava.impl.BlockImpl.class)
			return miniJava__Block((miniJava.Block) it);
		if (it.getClass() == miniJava.impl.BoolConstantImpl.class)
			return miniJava__BoolConstant__AS__miniJava__Statement((miniJava.BoolConstant) it);
		if (it.getClass() == miniJava.impl.DivisionImpl.class)
			return miniJava__Division__AS__miniJava__Statement((miniJava.Division) it);
		if (it.getClass() == miniJava.impl.EqualityImpl.class)
			return miniJava__Equality__AS__miniJava__Statement((miniJava.Equality) it);
		if (it.getClass() == miniJava.impl.ExpressionImpl.class)
			return miniJava__Expression__AS__miniJava__Statement((miniJava.Expression) it);
		if (it.getClass() == miniJava.impl.FieldAccessImpl.class)
			return miniJava__FieldAccess__AS__miniJava__Statement((miniJava.FieldAccess) it);
		if (it.getClass() == miniJava.impl.ForStatementImpl.class)
			return miniJava__ForStatement((miniJava.ForStatement) it);
		if (it.getClass() == miniJava.impl.IfStatementImpl.class)
			return miniJava__IfStatement((miniJava.IfStatement) it);
		if (it.getClass() == miniJava.impl.InequalityImpl.class)
			return miniJava__Inequality__AS__miniJava__Statement((miniJava.Inequality) it);
		if (it.getClass() == miniJava.impl.InferiorImpl.class)
			return miniJava__Inferior__AS__miniJava__Statement((miniJava.Inferior) it);
		if (it.getClass() == miniJava.impl.InferiorOrEqualImpl.class)
			return miniJava__InferiorOrEqual__AS__miniJava__Statement((miniJava.InferiorOrEqual) it);
		if (it.getClass() == miniJava.impl.IntConstantImpl.class)
			return miniJava__IntConstant__AS__miniJava__Statement((miniJava.IntConstant) it);
		if (it.getClass() == miniJava.impl.LoadImageImpl.class)
			return miniJava__LoadImage__AS__miniJava__Statement((miniJava.LoadImage) it);
		if (it.getClass() == miniJava.impl.MethodCallImpl.class)
			return miniJava__MethodCall__AS__miniJava__Statement((miniJava.MethodCall) it);
		if (it.getClass() == miniJava.impl.MinusImpl.class)
			return miniJava__Minus__AS__miniJava__Statement((miniJava.Minus) it);
		if (it.getClass() == miniJava.impl.ModuloImpl.class)
			return miniJava__Modulo__AS__miniJava__Statement((miniJava.Modulo) it);
		if (it.getClass() == miniJava.impl.MultiplicationImpl.class)
			return miniJava__Multiplication__AS__miniJava__Statement((miniJava.Multiplication) it);
		if (it.getClass() == miniJava.impl.NegImpl.class)
			return miniJava__Neg__AS__miniJava__Statement((miniJava.Neg) it);
		if (it.getClass() == miniJava.impl.NewArrayImpl.class)
			return miniJava__NewArray__AS__miniJava__Statement((miniJava.NewArray) it);
		if (it.getClass() == miniJava.impl.NewObjectImpl.class)
			return miniJava__NewObject__AS__miniJava__Statement((miniJava.NewObject) it);
		if (it.getClass() == miniJava.impl.NotImpl.class)
			return miniJava__Not__AS__miniJava__Statement((miniJava.Not) it);
		if (it.getClass() == miniJava.impl.NullImpl.class)
			return miniJava__Null__AS__miniJava__Statement((miniJava.Null) it);
		if (it.getClass() == miniJava.impl.OrImpl.class)
			return miniJava__Or__AS__miniJava__Statement((miniJava.Or) it);
		if (it.getClass() == miniJava.impl.PlusImpl.class)
			return miniJava__Plus__AS__miniJava__Statement((miniJava.Plus) it);
		if (it.getClass() == miniJava.impl.PrintStatementImpl.class)
			return miniJava__PrintStatement((miniJava.PrintStatement) it);
		if (it.getClass() == miniJava.impl.ReturnImpl.class)
			return miniJava__Return((miniJava.Return) it);
		if (it.getClass() == miniJava.impl.SqrtImpl.class)
			return miniJava__Sqrt__AS__miniJava__Statement((miniJava.Sqrt) it);
		if (it.getClass() == miniJava.impl.StringConstantImpl.class)
			return miniJava__StringConstant__AS__miniJava__Statement((miniJava.StringConstant) it);
		if (it.getClass() == miniJava.impl.SuperImpl.class)
			return miniJava__Super__AS__miniJava__Statement((miniJava.Super) it);
		if (it.getClass() == miniJava.impl.SuperiorImpl.class)
			return miniJava__Superior__AS__miniJava__Statement((miniJava.Superior) it);
		if (it.getClass() == miniJava.impl.SuperiorOrEqualImpl.class)
			return miniJava__SuperiorOrEqual__AS__miniJava__Statement((miniJava.SuperiorOrEqual) it);
		if (it.getClass() == miniJava.impl.SymbolRefImpl.class)
			return miniJava__SymbolRef__AS__miniJava__Statement((miniJava.SymbolRef) it);
		if (it.getClass() == miniJava.impl.ThisImpl.class)
			return miniJava__This__AS__miniJava__Statement((miniJava.This) it);
		if (it.getClass() == miniJava.impl.WhileStatementImpl.class)
			return miniJava__WhileStatement((miniJava.WhileStatement) it);
		if (it.getClass() == miniJava.impl.WriteImageImpl.class)
			return miniJava__WriteImage((miniJava.WriteImage) it);
		return miniJava__Statement(it);
	}
	default MiniJava__StringConstantT $(final miniJava.StringConstant it) {
		return miniJava__StringConstant(it);
	}
	default MiniJava__StringTypeRefT $(final miniJava.StringTypeRef it) {
		return miniJava__StringTypeRef(it);
	}
	default MiniJava__StringValueT $(final miniJava.StringValue it) {
		return miniJava__StringValue(it);
	}
	default MiniJava__SuperT $(final miniJava.Super it) {
		return miniJava__Super(it);
	}
	default MiniJava__SuperiorT $(final miniJava.Superior it) {
		return miniJava__Superior(it);
	}
	default MiniJava__SuperiorOrEqualT $(final miniJava.SuperiorOrEqual it) {
		return miniJava__SuperiorOrEqual(it);
	}
	default MiniJava__SymbolT $(final miniJava.Symbol it) {
		if (it.getClass() == miniJava.impl.ParameterImpl.class)
			return miniJava__Parameter((miniJava.Parameter) it);
		if (it.getClass() == miniJava.impl.VariableDeclarationImpl.class)
			return miniJava__VariableDeclaration__AS__miniJava__Symbol((miniJava.VariableDeclaration) it);
		return miniJava__Symbol(it);
	}
	default MiniJava__SymbolBindingT $(final miniJava.SymbolBinding it) {
		return miniJava__SymbolBinding(it);
	}
	default MiniJava__SymbolRefT $(final miniJava.SymbolRef it) {
		return miniJava__SymbolRef(it);
	}
	default MiniJava__ThisT $(final miniJava.This it) {
		return miniJava__This(it);
	}
	default MiniJava__TypeDeclarationT $(final miniJava.TypeDeclaration it) {
		if (it.getClass() == miniJava.impl.ClazzImpl.class)
			return miniJava__Clazz((miniJava.Clazz) it);
		if (it.getClass() == miniJava.impl.InterfaceImpl.class)
			return miniJava__Interface((miniJava.Interface) it);
		return miniJava__TypeDeclaration(it);
	}
	default MiniJava__TypeRefT $(final miniJava.TypeRef it) {
		if (it.getClass() == miniJava.impl.ArrayTypeRefImpl.class)
			return miniJava__ArrayTypeRef((miniJava.ArrayTypeRef) it);
		if (it.getClass() == miniJava.impl.BooleanTypeRefImpl.class)
			return miniJava__BooleanTypeRef((miniJava.BooleanTypeRef) it);
		if (it.getClass() == miniJava.impl.ClassRefImpl.class)
			return miniJava__ClassRef((miniJava.ClassRef) it);
		if (it.getClass() == miniJava.impl.IntegerTypeRefImpl.class)
			return miniJava__IntegerTypeRef((miniJava.IntegerTypeRef) it);
		if (it.getClass() == miniJava.impl.SingleTypeRefImpl.class)
			return miniJava__SingleTypeRef((miniJava.SingleTypeRef) it);
		if (it.getClass() == miniJava.impl.StringTypeRefImpl.class)
			return miniJava__StringTypeRef((miniJava.StringTypeRef) it);
		if (it.getClass() == miniJava.impl.VoidTypeRefImpl.class)
			return miniJava__VoidTypeRef((miniJava.VoidTypeRef) it);
		return miniJava__TypeRef(it);
	}
	default MiniJava__TypedDeclarationT $(final miniJava.TypedDeclaration it) {
		if (it.getClass() == miniJava.impl.FieldImpl.class)
			return miniJava__Field((miniJava.Field) it);
		if (it.getClass() == miniJava.impl.MemberImpl.class)
			return miniJava__Member((miniJava.Member) it);
		if (it.getClass() == miniJava.impl.MethodImpl.class)
			return miniJava__Method((miniJava.Method) it);
		if (it.getClass() == miniJava.impl.ParameterImpl.class)
			return miniJava__Parameter((miniJava.Parameter) it);
		if (it.getClass() == miniJava.impl.SymbolImpl.class)
			return miniJava__Symbol((miniJava.Symbol) it);
		if (it.getClass() == miniJava.impl.VariableDeclarationImpl.class)
			return miniJava__VariableDeclaration__AS__miniJava__Symbol((miniJava.VariableDeclaration) it);
		return miniJava__TypedDeclaration(it);
	}
	default MiniJava__ValueT $(final miniJava.Value it) {
		if (it.getClass() == miniJava.impl.ArrayRefValueImpl.class)
			return miniJava__ArrayRefValue((miniJava.ArrayRefValue) it);
		if (it.getClass() == miniJava.impl.BooleanValueImpl.class)
			return miniJava__BooleanValue((miniJava.BooleanValue) it);
		if (it.getClass() == miniJava.impl.IntegerValueImpl.class)
			return miniJava__IntegerValue((miniJava.IntegerValue) it);
		if (it.getClass() == miniJava.impl.NullValueImpl.class)
			return miniJava__NullValue((miniJava.NullValue) it);
		if (it.getClass() == miniJava.impl.ObjectRefValueImpl.class)
			return miniJava__ObjectRefValue((miniJava.ObjectRefValue) it);
		if (it.getClass() == miniJava.impl.StringValueImpl.class)
			return miniJava__StringValue((miniJava.StringValue) it);
		return miniJava__Value(it);
	}
	default MiniJava__VariableDeclarationT $(final miniJava.VariableDeclaration it) {
		return miniJava__VariableDeclaration(it);
	}
	default MiniJava__VoidTypeRefT $(final miniJava.VoidTypeRef it) {
		return miniJava__VoidTypeRef(it);
	}
	default MiniJava__WhileStatementT $(final miniJava.WhileStatement it) {
		return miniJava__WhileStatement(it);
	}
	default MiniJava__WriteImageT $(final miniJava.WriteImage it) {
		return miniJava__WriteImage(it);
	}
}
