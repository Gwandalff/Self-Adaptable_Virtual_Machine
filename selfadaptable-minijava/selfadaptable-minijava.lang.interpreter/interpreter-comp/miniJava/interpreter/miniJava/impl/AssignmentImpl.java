package miniJava.interpreter.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.interpreter.miniJava.ArrayAccess;
import miniJava.interpreter.miniJava.ArrayInstance;
import miniJava.interpreter.miniJava.ArrayRefValue;
import miniJava.interpreter.miniJava.Assignee;
import miniJava.interpreter.miniJava.Assignment;
import miniJava.interpreter.miniJava.Context;
import miniJava.interpreter.miniJava.Expression;
import miniJava.interpreter.miniJava.Field;
import miniJava.interpreter.miniJava.FieldAccess;
import miniJava.interpreter.miniJava.FieldBinding;
import miniJava.interpreter.miniJava.IntegerValue;
import miniJava.interpreter.miniJava.MiniJavaFactory;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.ObjectInstance;
import miniJava.interpreter.miniJava.ObjectRefValue;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.Symbol;
import miniJava.interpreter.miniJava.SymbolBinding;
import miniJava.interpreter.miniJava.SymbolRef;
import miniJava.interpreter.miniJava.Value;
import miniJava.interpreter.miniJava.VariableDeclaration;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecoretools.ale.compiler.lib.CollectionService;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;

public class AssignmentImpl extends StatementImpl implements Assignment {
	protected Assignee assignee;

	protected Expression value;

	protected AssignmentImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.ASSIGNMENT;
	}

	public Assignee getAssignee() {
		return assignee;
	}

	public NotificationChain basicSetAssignee(Assignee newAssignee, NotificationChain msgs) {
		Assignee oldAssignee = assignee;
		assignee = newAssignee;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.ASSIGNMENT__ASSIGNEE, oldAssignee, newAssignee);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setAssignee(Assignee newAssignee) {
		if (newAssignee != assignee) {
			NotificationChain msgs = null;
			if (assignee != null)
				msgs = ((InternalEObject) assignee).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.ASSIGNMENT__ASSIGNEE, null, msgs);
			if (newAssignee != null)
				msgs = ((InternalEObject) newAssignee).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.ASSIGNMENT__ASSIGNEE, null, msgs);
			msgs = basicSetAssignee(newAssignee, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.ASSIGNMENT__ASSIGNEE, newAssignee, newAssignee));
	}

	public Expression getValue() {
		return value;
	}

	public NotificationChain basicSetValue(Expression newValue, NotificationChain msgs) {
		Expression oldValue = value;
		value = newValue;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.ASSIGNMENT__VALUE, oldValue, newValue);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setValue(Expression newValue) {
		if (newValue != value) {
			NotificationChain msgs = null;
			if (value != null)
				msgs = ((InternalEObject) value).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.ASSIGNMENT__VALUE, null, msgs);
			if (newValue != null)
				msgs = ((InternalEObject) newValue).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.ASSIGNMENT__VALUE, null, msgs);
			msgs = basicSetValue(newValue, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.ASSIGNMENT__VALUE, newValue, newValue));
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.ASSIGNMENT__ASSIGNEE :
				return basicSetAssignee(null, msgs);
			case MiniJavaPackage.ASSIGNMENT__VALUE :
				return basicSetValue(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.ASSIGNMENT__ASSIGNEE :
				return getAssignee();
			case MiniJavaPackage.ASSIGNMENT__VALUE :
				return getValue();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.ASSIGNMENT__ASSIGNEE :
				setAssignee((Assignee) newValue);
				return;
			case MiniJavaPackage.ASSIGNMENT__VALUE :
				setValue((Expression) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.ASSIGNMENT__ASSIGNEE :
				setAssignee((Assignee) null);
				return;
			case MiniJavaPackage.ASSIGNMENT__VALUE :
				setValue((Expression) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.ASSIGNMENT__ASSIGNEE :
				return assignee != null;
			case MiniJavaPackage.ASSIGNMENT__VALUE :
				return value != null;
		}
		return super.eIsSet(featureID);
	}

	public void evaluateStatement(State state) {
		Context context = ((Context) (((State) (state)).findCurrentContext()));
		Value right = ((Value) (((Expression) (this.getValue())).evaluateExpression((State) (state))));
		Assignee assignee = ((Assignee) (this.getAssignee()));
		if (assignee instanceof SymbolRef) {
			SymbolRef assigneeSymbolRef = ((SymbolRef) (assignee));
			SymbolBinding existingBinding = ((SymbolBinding) (((Context) (context)).findBinding((Symbol) (assigneeSymbolRef.getSymbol()))));
			existingBinding.setValue(right);
		}
		else {
			if (assignee instanceof VariableDeclaration) {
				VariableDeclaration assigneeVariableDeclaration = ((VariableDeclaration) (assignee));
				SymbolBinding binding = ((SymbolBinding) (MiniJavaFactory.eINSTANCE.createSymbolBinding()));
				binding.setSymbol(assigneeVariableDeclaration);
				binding.setValue(right);
				context.getBindings().add(binding);
			}
			else {
				if (assignee instanceof FieldAccess) {
					FieldAccess assigneeFieldAccess = ((FieldAccess) (assignee));
					Field f = ((Field) (assigneeFieldAccess.getField()));
					ObjectRefValue realReceiverValue = ((ObjectRefValue) (((ObjectRefValue) (((Expression) (assigneeFieldAccess.getReceiver())).evaluateExpression((State) (state))))));
					ObjectInstance realReceiver = ((ObjectInstance) (realReceiverValue.getInstance()));
					FieldBinding existingBinding = ((FieldBinding) (CollectionService.head(CollectionService.select(realReceiver.getFieldbindings(), (x) -> EqualService.equals((x.getField()), (f))))));
					if (EqualService.equals((existingBinding), (null))) {
						FieldBinding binding = ((FieldBinding) (MiniJavaFactory.eINSTANCE.createFieldBinding()));
						binding.setField(f);
						binding.setValue(right);
						realReceiver.getFieldbindings().add(binding);
					}
					else {
						existingBinding.setValue(right);
					}
				}
				else {
					if (assignee instanceof ArrayAccess) {
						ArrayAccess assigneeArrayAccess = ((ArrayAccess) (assignee));
						ArrayRefValue arrayRefValue = ((ArrayRefValue) (((ArrayRefValue) (((Expression) (assigneeArrayAccess.getObject())).evaluateExpression((State) (state))))));
						ArrayInstance array = ((ArrayInstance) (arrayRefValue.getInstance()));
						IntegerValue integerValue = ((IntegerValue) (((IntegerValue) (((Expression) (assigneeArrayAccess.getIndex())).evaluateExpression((State) (state))))));
						int index = ((int) (integerValue.getValue()));
						CollectionService.set(array.getValue(), index, right);
					}
					else {
					}
				}
			}
		}
	}
}
