package visitor;

import java.lang.Object;
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

public interface VisitorInterface {
	Object visitminiJava__Program(Program it);

	Object visitminiJava__Import(Import it);

	Object visitminiJava__TypeDeclaration(TypeDeclaration it);

	Object visitminiJava__Clazz(Clazz it);

	Object visitminiJava__Interface(Interface it);

	Object visitminiJava__Member(Member it);

	Object visitminiJava__Method(Method it);

	Object visitminiJava__Parameter(Parameter it);

	Object visitminiJava__Field(Field it);

	Object visitminiJava__Block(Block it);

	Object visitminiJava__Statement(Statement it);

	Object visitminiJava__PrintStatement(PrintStatement it);

	Object visitminiJava__LoadImage(LoadImage it);

	Object visitminiJava__WriteImage(WriteImage it);

	Object visitminiJava__Return(Return it);

	Object visitminiJava__IfStatement(IfStatement it);

	Object visitminiJava__WhileStatement(WhileStatement it);

	Object visitminiJava__ForStatement(ForStatement it);

	Object visitminiJava__TypeRef(TypeRef it);

	Object visitminiJava__SingleTypeRef(SingleTypeRef it);

	Object visitminiJava__ClassRef(ClassRef it);

	Object visitminiJava__NamedElement(NamedElement it);

	Object visitminiJava__TypedDeclaration(TypedDeclaration it);

	Object visitminiJava__Symbol(Symbol it);

	Object visitminiJava__VariableDeclaration(VariableDeclaration it);

	Object visitminiJava__Assignment(Assignment it);

	Object visitminiJava__Assignee(Assignee it);

	Object visitminiJava__Expression(Expression it);

	Object visitminiJava__ArrayTypeRef(ArrayTypeRef it);

	Object visitminiJava__IntegerTypeRef(IntegerTypeRef it);

	Object visitminiJava__BooleanTypeRef(BooleanTypeRef it);

	Object visitminiJava__StringTypeRef(StringTypeRef it);

	Object visitminiJava__VoidTypeRef(VoidTypeRef it);

	Object visitminiJava__Or(Or it);

	Object visitminiJava__And(And it);

	Object visitminiJava__Equality(Equality it);

	Object visitminiJava__Inequality(Inequality it);

	Object visitminiJava__SuperiorOrEqual(SuperiorOrEqual it);

	Object visitminiJava__InferiorOrEqual(InferiorOrEqual it);

	Object visitminiJava__Superior(Superior it);

	Object visitminiJava__Inferior(Inferior it);

	Object visitminiJava__Plus(Plus it);

	Object visitminiJava__Minus(Minus it);

	Object visitminiJava__Multiplication(Multiplication it);

	Object visitminiJava__Division(Division it);

	Object visitminiJava__ArrayAccess(ArrayAccess it);

	Object visitminiJava__ArrayLength(ArrayLength it);

	Object visitminiJava__Not(Not it);

	Object visitminiJava__Neg(Neg it);

	Object visitminiJava__FieldAccess(FieldAccess it);

	Object visitminiJava__MethodCall(MethodCall it);

	Object visitminiJava__StringConstant(StringConstant it);

	Object visitminiJava__IntConstant(IntConstant it);

	Object visitminiJava__BoolConstant(BoolConstant it);

	Object visitminiJava__This(This it);

	Object visitminiJava__Super(Super it);

	Object visitminiJava__Null(Null it);

	Object visitminiJava__NewObject(NewObject it);

	Object visitminiJava__NewArray(NewArray it);

	Object visitminiJava__SymbolRef(SymbolRef it);

	Object visitminiJava__Context(Context it);

	Object visitminiJava__Value(Value it);

	Object visitminiJava__IntegerValue(IntegerValue it);

	Object visitminiJava__SymbolBinding(SymbolBinding it);

	Object visitminiJava__FieldBinding(FieldBinding it);

	Object visitminiJava__StringValue(StringValue it);

	Object visitminiJava__BooleanValue(BooleanValue it);

	Object visitminiJava__OutputStream(OutputStream it);

	Object visitminiJava__State(State it);

	Object visitminiJava__Frame(Frame it);

	Object visitminiJava__NullValue(NullValue it);

	Object visitminiJava__NewCall(NewCall it);

	Object visitminiJava__MethodCall2(MethodCall2 it);

	Object visitminiJava__ObjectInstance(ObjectInstance it);

	Object visitminiJava__ArrayInstance(ArrayInstance it);

	Object visitminiJava__ObjectRefValue(ObjectRefValue it);

	Object visitminiJava__ArrayRefValue(ArrayRefValue it);

	Object visitminiJava__SymbolToSymbolBindingMap(SymbolToSymbolBindingMap it);

	Object visitminiJava__ClazzToMethodMap(ClazzToMethodMap it);

	Object visitminiJava__Modulo(Modulo it);

	Object visitminiJava__Sqrt(Sqrt it);
}
