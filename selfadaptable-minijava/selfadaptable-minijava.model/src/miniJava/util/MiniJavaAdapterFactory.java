/**
 */
package miniJava.util;

import java.util.Map;

import miniJava.*;

import org.eclipse.emf.common.notify.Adapter;
import org.eclipse.emf.common.notify.Notifier;

import org.eclipse.emf.common.notify.impl.AdapterFactoryImpl;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * The <b>Adapter Factory</b> for the model.
 * It provides an adapter <code>createXXX</code> method for each class of the model.
 * <!-- end-user-doc -->
 * @see miniJava.MiniJavaPackage
 * @generated
 */
public class MiniJavaAdapterFactory extends AdapterFactoryImpl {
	/**
	 * The cached model package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected static MiniJavaPackage modelPackage;

	/**
	 * Creates an instance of the adapter factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MiniJavaAdapterFactory() {
		if (modelPackage == null) {
			modelPackage = MiniJavaPackage.eINSTANCE;
		}
	}

	/**
	 * Returns whether this factory is applicable for the type of the object.
	 * <!-- begin-user-doc -->
	 * This implementation returns <code>true</code> if the object is either the model's package or is an instance object of the model.
	 * <!-- end-user-doc -->
	 * @return whether this factory is applicable for the type of the object.
	 * @generated
	 */
	@Override
	public boolean isFactoryForType(Object object) {
		if (object == modelPackage) {
			return true;
		}
		if (object instanceof EObject) {
			return ((EObject)object).eClass().getEPackage() == modelPackage;
		}
		return false;
	}

	/**
	 * The switch that delegates to the <code>createXXX</code> methods.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MiniJavaSwitch<Adapter> modelSwitch =
		new MiniJavaSwitch<Adapter>() {
			@Override
			public Adapter caseProgram(Program object) {
				return createProgramAdapter();
			}
			@Override
			public Adapter caseImport(Import object) {
				return createImportAdapter();
			}
			@Override
			public Adapter caseTypeDeclaration(TypeDeclaration object) {
				return createTypeDeclarationAdapter();
			}
			@Override
			public Adapter caseClazz(Clazz object) {
				return createClazzAdapter();
			}
			@Override
			public Adapter caseInterface(Interface object) {
				return createInterfaceAdapter();
			}
			@Override
			public Adapter caseMember(Member object) {
				return createMemberAdapter();
			}
			@Override
			public Adapter caseMethod(Method object) {
				return createMethodAdapter();
			}
			@Override
			public Adapter caseParameter(Parameter object) {
				return createParameterAdapter();
			}
			@Override
			public Adapter caseField(Field object) {
				return createFieldAdapter();
			}
			@Override
			public Adapter caseBlock(Block object) {
				return createBlockAdapter();
			}
			@Override
			public Adapter caseStatement(Statement object) {
				return createStatementAdapter();
			}
			@Override
			public Adapter casePrintStatement(PrintStatement object) {
				return createPrintStatementAdapter();
			}
			@Override
			public Adapter caseReturn(Return object) {
				return createReturnAdapter();
			}
			@Override
			public Adapter caseIfStatement(IfStatement object) {
				return createIfStatementAdapter();
			}
			@Override
			public Adapter caseWhileStatement(WhileStatement object) {
				return createWhileStatementAdapter();
			}
			@Override
			public Adapter caseForStatement(ForStatement object) {
				return createForStatementAdapter();
			}
			@Override
			public Adapter caseTypeRef(TypeRef object) {
				return createTypeRefAdapter();
			}
			@Override
			public Adapter caseSingleTypeRef(SingleTypeRef object) {
				return createSingleTypeRefAdapter();
			}
			@Override
			public Adapter caseClassRef(ClassRef object) {
				return createClassRefAdapter();
			}
			@Override
			public Adapter caseNamedElement(NamedElement object) {
				return createNamedElementAdapter();
			}
			@Override
			public Adapter caseTypedDeclaration(TypedDeclaration object) {
				return createTypedDeclarationAdapter();
			}
			@Override
			public Adapter caseSymbol(Symbol object) {
				return createSymbolAdapter();
			}
			@Override
			public Adapter caseVariableDeclaration(VariableDeclaration object) {
				return createVariableDeclarationAdapter();
			}
			@Override
			public Adapter caseAssignment(Assignment object) {
				return createAssignmentAdapter();
			}
			@Override
			public Adapter caseAssignee(Assignee object) {
				return createAssigneeAdapter();
			}
			@Override
			public Adapter caseExpression(Expression object) {
				return createExpressionAdapter();
			}
			@Override
			public Adapter caseArrayTypeRef(ArrayTypeRef object) {
				return createArrayTypeRefAdapter();
			}
			@Override
			public Adapter caseIntegerTypeRef(IntegerTypeRef object) {
				return createIntegerTypeRefAdapter();
			}
			@Override
			public Adapter caseBooleanTypeRef(BooleanTypeRef object) {
				return createBooleanTypeRefAdapter();
			}
			@Override
			public Adapter caseStringTypeRef(StringTypeRef object) {
				return createStringTypeRefAdapter();
			}
			@Override
			public Adapter caseVoidTypeRef(VoidTypeRef object) {
				return createVoidTypeRefAdapter();
			}
			@Override
			public Adapter caseOr(Or object) {
				return createOrAdapter();
			}
			@Override
			public Adapter caseAnd(And object) {
				return createAndAdapter();
			}
			@Override
			public Adapter caseEquality(Equality object) {
				return createEqualityAdapter();
			}
			@Override
			public Adapter caseInequality(Inequality object) {
				return createInequalityAdapter();
			}
			@Override
			public Adapter caseSuperiorOrEqual(SuperiorOrEqual object) {
				return createSuperiorOrEqualAdapter();
			}
			@Override
			public Adapter caseInferiorOrEqual(InferiorOrEqual object) {
				return createInferiorOrEqualAdapter();
			}
			@Override
			public Adapter caseSuperior(Superior object) {
				return createSuperiorAdapter();
			}
			@Override
			public Adapter caseInferior(Inferior object) {
				return createInferiorAdapter();
			}
			@Override
			public Adapter casePlus(Plus object) {
				return createPlusAdapter();
			}
			@Override
			public Adapter caseMinus(Minus object) {
				return createMinusAdapter();
			}
			@Override
			public Adapter caseMultiplication(Multiplication object) {
				return createMultiplicationAdapter();
			}
			@Override
			public Adapter caseDivision(Division object) {
				return createDivisionAdapter();
			}
			@Override
			public Adapter caseArrayAccess(ArrayAccess object) {
				return createArrayAccessAdapter();
			}
			@Override
			public Adapter caseArrayLength(ArrayLength object) {
				return createArrayLengthAdapter();
			}
			@Override
			public Adapter caseNot(Not object) {
				return createNotAdapter();
			}
			@Override
			public Adapter caseNeg(Neg object) {
				return createNegAdapter();
			}
			@Override
			public Adapter caseFieldAccess(FieldAccess object) {
				return createFieldAccessAdapter();
			}
			@Override
			public Adapter caseMethodCall(MethodCall object) {
				return createMethodCallAdapter();
			}
			@Override
			public Adapter caseStringConstant(StringConstant object) {
				return createStringConstantAdapter();
			}
			@Override
			public Adapter caseIntConstant(IntConstant object) {
				return createIntConstantAdapter();
			}
			@Override
			public Adapter caseBoolConstant(BoolConstant object) {
				return createBoolConstantAdapter();
			}
			@Override
			public Adapter caseThis(This object) {
				return createThisAdapter();
			}
			@Override
			public Adapter caseSuper(Super object) {
				return createSuperAdapter();
			}
			@Override
			public Adapter caseNull(Null object) {
				return createNullAdapter();
			}
			@Override
			public Adapter caseNewObject(NewObject object) {
				return createNewObjectAdapter();
			}
			@Override
			public Adapter caseNewArray(NewArray object) {
				return createNewArrayAdapter();
			}
			@Override
			public Adapter caseSymbolRef(SymbolRef object) {
				return createSymbolRefAdapter();
			}
			@Override
			public Adapter caseContext(Context object) {
				return createContextAdapter();
			}
			@Override
			public Adapter caseValue(Value object) {
				return createValueAdapter();
			}
			@Override
			public Adapter caseIntegerValue(IntegerValue object) {
				return createIntegerValueAdapter();
			}
			@Override
			public Adapter caseSymbolBinding(SymbolBinding object) {
				return createSymbolBindingAdapter();
			}
			@Override
			public Adapter caseFieldBinding(FieldBinding object) {
				return createFieldBindingAdapter();
			}
			@Override
			public Adapter caseStringValue(StringValue object) {
				return createStringValueAdapter();
			}
			@Override
			public Adapter caseBooleanValue(BooleanValue object) {
				return createBooleanValueAdapter();
			}
			@Override
			public Adapter caseOutputStream(OutputStream object) {
				return createOutputStreamAdapter();
			}
			@Override
			public Adapter caseState(State object) {
				return createStateAdapter();
			}
			@Override
			public Adapter caseFrame(Frame object) {
				return createFrameAdapter();
			}
			@Override
			public Adapter caseNullValue(NullValue object) {
				return createNullValueAdapter();
			}
			@Override
			public Adapter caseCall(Call object) {
				return createCallAdapter();
			}
			@Override
			public Adapter caseNewCall(NewCall object) {
				return createNewCallAdapter();
			}
			@Override
			public Adapter caseMethodCall2(MethodCall2 object) {
				return createMethodCall2Adapter();
			}
			@Override
			public Adapter caseObjectInstance(ObjectInstance object) {
				return createObjectInstanceAdapter();
			}
			@Override
			public Adapter caseArrayInstance(ArrayInstance object) {
				return createArrayInstanceAdapter();
			}
			@Override
			public Adapter caseObjectRefValue(ObjectRefValue object) {
				return createObjectRefValueAdapter();
			}
			@Override
			public Adapter caseArrayRefValue(ArrayRefValue object) {
				return createArrayRefValueAdapter();
			}
			@Override
			public Adapter caseSymbolToSymbolBindingMap(Map.Entry<Symbol, SymbolBinding> object) {
				return createSymbolToSymbolBindingMapAdapter();
			}
			@Override
			public Adapter caseClazzToMethodMap(Map.Entry<Clazz, Method> object) {
				return createClazzToMethodMapAdapter();
			}
			@Override
			public Adapter caseModulo(Modulo object) {
				return createModuloAdapter();
			}
			@Override
			public Adapter defaultCase(EObject object) {
				return createEObjectAdapter();
			}
		};

	/**
	 * Creates an adapter for the <code>target</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param target the object to adapt.
	 * @return the adapter for the <code>target</code>.
	 * @generated
	 */
	@Override
	public Adapter createAdapter(Notifier target) {
		return modelSwitch.doSwitch((EObject)target);
	}


	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Program <em>Program</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Program
	 * @generated
	 */
	public Adapter createProgramAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Import <em>Import</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Import
	 * @generated
	 */
	public Adapter createImportAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.TypeDeclaration <em>Type Declaration</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.TypeDeclaration
	 * @generated
	 */
	public Adapter createTypeDeclarationAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Clazz <em>Clazz</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Clazz
	 * @generated
	 */
	public Adapter createClazzAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Interface <em>Interface</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Interface
	 * @generated
	 */
	public Adapter createInterfaceAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Member <em>Member</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Member
	 * @generated
	 */
	public Adapter createMemberAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Method <em>Method</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Method
	 * @generated
	 */
	public Adapter createMethodAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Parameter <em>Parameter</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Parameter
	 * @generated
	 */
	public Adapter createParameterAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Field <em>Field</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Field
	 * @generated
	 */
	public Adapter createFieldAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Block <em>Block</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Block
	 * @generated
	 */
	public Adapter createBlockAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Statement <em>Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Statement
	 * @generated
	 */
	public Adapter createStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.PrintStatement <em>Print Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.PrintStatement
	 * @generated
	 */
	public Adapter createPrintStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Return <em>Return</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Return
	 * @generated
	 */
	public Adapter createReturnAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.IfStatement <em>If Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.IfStatement
	 * @generated
	 */
	public Adapter createIfStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.WhileStatement <em>While Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.WhileStatement
	 * @generated
	 */
	public Adapter createWhileStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.ForStatement <em>For Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.ForStatement
	 * @generated
	 */
	public Adapter createForStatementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.TypeRef <em>Type Ref</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.TypeRef
	 * @generated
	 */
	public Adapter createTypeRefAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.SingleTypeRef <em>Single Type Ref</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.SingleTypeRef
	 * @generated
	 */
	public Adapter createSingleTypeRefAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.ClassRef <em>Class Ref</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.ClassRef
	 * @generated
	 */
	public Adapter createClassRefAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.NamedElement <em>Named Element</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.NamedElement
	 * @generated
	 */
	public Adapter createNamedElementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.TypedDeclaration <em>Typed Declaration</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.TypedDeclaration
	 * @generated
	 */
	public Adapter createTypedDeclarationAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Symbol <em>Symbol</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Symbol
	 * @generated
	 */
	public Adapter createSymbolAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.VariableDeclaration <em>Variable Declaration</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.VariableDeclaration
	 * @generated
	 */
	public Adapter createVariableDeclarationAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Assignment <em>Assignment</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Assignment
	 * @generated
	 */
	public Adapter createAssignmentAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Assignee <em>Assignee</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Assignee
	 * @generated
	 */
	public Adapter createAssigneeAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Expression <em>Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Expression
	 * @generated
	 */
	public Adapter createExpressionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.ArrayTypeRef <em>Array Type Ref</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.ArrayTypeRef
	 * @generated
	 */
	public Adapter createArrayTypeRefAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.IntegerTypeRef <em>Integer Type Ref</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.IntegerTypeRef
	 * @generated
	 */
	public Adapter createIntegerTypeRefAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.BooleanTypeRef <em>Boolean Type Ref</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.BooleanTypeRef
	 * @generated
	 */
	public Adapter createBooleanTypeRefAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.StringTypeRef <em>String Type Ref</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.StringTypeRef
	 * @generated
	 */
	public Adapter createStringTypeRefAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.VoidTypeRef <em>Void Type Ref</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.VoidTypeRef
	 * @generated
	 */
	public Adapter createVoidTypeRefAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Or <em>Or</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Or
	 * @generated
	 */
	public Adapter createOrAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.And <em>And</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.And
	 * @generated
	 */
	public Adapter createAndAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Equality <em>Equality</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Equality
	 * @generated
	 */
	public Adapter createEqualityAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Inequality <em>Inequality</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Inequality
	 * @generated
	 */
	public Adapter createInequalityAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.SuperiorOrEqual <em>Superior Or Equal</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.SuperiorOrEqual
	 * @generated
	 */
	public Adapter createSuperiorOrEqualAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.InferiorOrEqual <em>Inferior Or Equal</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.InferiorOrEqual
	 * @generated
	 */
	public Adapter createInferiorOrEqualAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Superior <em>Superior</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Superior
	 * @generated
	 */
	public Adapter createSuperiorAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Inferior <em>Inferior</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Inferior
	 * @generated
	 */
	public Adapter createInferiorAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Plus <em>Plus</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Plus
	 * @generated
	 */
	public Adapter createPlusAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Minus <em>Minus</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Minus
	 * @generated
	 */
	public Adapter createMinusAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Multiplication <em>Multiplication</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Multiplication
	 * @generated
	 */
	public Adapter createMultiplicationAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Division <em>Division</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Division
	 * @generated
	 */
	public Adapter createDivisionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.ArrayAccess <em>Array Access</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.ArrayAccess
	 * @generated
	 */
	public Adapter createArrayAccessAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.ArrayLength <em>Array Length</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.ArrayLength
	 * @generated
	 */
	public Adapter createArrayLengthAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Not <em>Not</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Not
	 * @generated
	 */
	public Adapter createNotAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Neg <em>Neg</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Neg
	 * @generated
	 */
	public Adapter createNegAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.FieldAccess <em>Field Access</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.FieldAccess
	 * @generated
	 */
	public Adapter createFieldAccessAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.MethodCall <em>Method Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.MethodCall
	 * @generated
	 */
	public Adapter createMethodCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.StringConstant <em>String Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.StringConstant
	 * @generated
	 */
	public Adapter createStringConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.IntConstant <em>Int Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.IntConstant
	 * @generated
	 */
	public Adapter createIntConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.BoolConstant <em>Bool Constant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.BoolConstant
	 * @generated
	 */
	public Adapter createBoolConstantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.This <em>This</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.This
	 * @generated
	 */
	public Adapter createThisAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Super <em>Super</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Super
	 * @generated
	 */
	public Adapter createSuperAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Null <em>Null</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Null
	 * @generated
	 */
	public Adapter createNullAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.NewObject <em>New Object</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.NewObject
	 * @generated
	 */
	public Adapter createNewObjectAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.NewArray <em>New Array</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.NewArray
	 * @generated
	 */
	public Adapter createNewArrayAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.SymbolRef <em>Symbol Ref</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.SymbolRef
	 * @generated
	 */
	public Adapter createSymbolRefAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Context <em>Context</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Context
	 * @generated
	 */
	public Adapter createContextAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Value <em>Value</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Value
	 * @generated
	 */
	public Adapter createValueAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.IntegerValue <em>Integer Value</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.IntegerValue
	 * @generated
	 */
	public Adapter createIntegerValueAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.SymbolBinding <em>Symbol Binding</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.SymbolBinding
	 * @generated
	 */
	public Adapter createSymbolBindingAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.FieldBinding <em>Field Binding</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.FieldBinding
	 * @generated
	 */
	public Adapter createFieldBindingAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.StringValue <em>String Value</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.StringValue
	 * @generated
	 */
	public Adapter createStringValueAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.BooleanValue <em>Boolean Value</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.BooleanValue
	 * @generated
	 */
	public Adapter createBooleanValueAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.OutputStream <em>Output Stream</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.OutputStream
	 * @generated
	 */
	public Adapter createOutputStreamAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.State <em>State</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.State
	 * @generated
	 */
	public Adapter createStateAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Frame <em>Frame</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Frame
	 * @generated
	 */
	public Adapter createFrameAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.NullValue <em>Null Value</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.NullValue
	 * @generated
	 */
	public Adapter createNullValueAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Call <em>Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Call
	 * @generated
	 */
	public Adapter createCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.NewCall <em>New Call</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.NewCall
	 * @generated
	 */
	public Adapter createNewCallAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.MethodCall2 <em>Method Call2</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.MethodCall2
	 * @generated
	 */
	public Adapter createMethodCall2Adapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.ObjectInstance <em>Object Instance</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.ObjectInstance
	 * @generated
	 */
	public Adapter createObjectInstanceAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.ArrayInstance <em>Array Instance</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.ArrayInstance
	 * @generated
	 */
	public Adapter createArrayInstanceAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.ObjectRefValue <em>Object Ref Value</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.ObjectRefValue
	 * @generated
	 */
	public Adapter createObjectRefValueAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.ArrayRefValue <em>Array Ref Value</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.ArrayRefValue
	 * @generated
	 */
	public Adapter createArrayRefValueAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link java.util.Map.Entry <em>Symbol To Symbol Binding Map</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see java.util.Map.Entry
	 * @generated
	 */
	public Adapter createSymbolToSymbolBindingMapAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link java.util.Map.Entry <em>Clazz To Method Map</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see java.util.Map.Entry
	 * @generated
	 */
	public Adapter createClazzToMethodMapAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link miniJava.Modulo <em>Modulo</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see miniJava.Modulo
	 * @generated
	 */
	public Adapter createModuloAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for the default case.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @generated
	 */
	public Adapter createEObjectAdapter() {
		return null;
	}

} //MiniJavaAdapterFactory
