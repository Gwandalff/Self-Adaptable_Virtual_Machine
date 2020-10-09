package miniJava.interpreter.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.util.Collection;
import miniJava.interpreter.miniJava.Clazz;
import miniJava.interpreter.miniJava.Context;
import miniJava.interpreter.miniJava.Expression;
import miniJava.interpreter.miniJava.Method;
import miniJava.interpreter.miniJava.MethodCall;
import miniJava.interpreter.miniJava.MethodCall2;
import miniJava.interpreter.miniJava.MiniJavaFactory;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.ObjectInstance;
import miniJava.interpreter.miniJava.ObjectRefValue;
import miniJava.interpreter.miniJava.Parameter;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.SymbolBinding;
import miniJava.interpreter.miniJava.Value;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;
import org.eclipse.emf.ecoretools.ale.compiler.lib.CollectionService;

public class MethodCallImpl extends ExpressionImpl implements MethodCall {
	protected Expression receiver;

	protected Method method;

	protected EList<Expression> args;

	protected MethodCallImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.METHOD_CALL;
	}

	public Expression getReceiver() {
		return receiver;
	}

	public NotificationChain basicSetReceiver(Expression newReceiver, NotificationChain msgs) {
		Expression oldReceiver = receiver;
		receiver = newReceiver;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.METHOD_CALL__RECEIVER, oldReceiver, newReceiver);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setReceiver(Expression newReceiver) {
		if (newReceiver != receiver) {
			NotificationChain msgs = null;
			if (receiver != null)
				msgs = ((InternalEObject) receiver).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.METHOD_CALL__RECEIVER, null, msgs);
			if (newReceiver != null)
				msgs = ((InternalEObject) newReceiver).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.METHOD_CALL__RECEIVER, null, msgs);
			msgs = basicSetReceiver(newReceiver, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.METHOD_CALL__RECEIVER, newReceiver, newReceiver));
	}

	public Method getMethod() {
		if (method != null && method.eIsProxy()) {
			InternalEObject oldMethod = (InternalEObject) method;
			method = (Method) eResolveProxy(oldMethod);
			if (method != oldMethod) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.METHOD_CALL__METHOD, oldMethod, method));
			}
		}
		return method;
	}

	public Method basicGetMethod() {
		return method;
	}

	public void setMethod(Method newMethod) {
		Method oldMethod = method;
		method = newMethod;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.METHOD_CALL__METHOD, oldMethod, method));
	}

	public EList<Expression> getArgs() {
		if (args == null) {
			args = new EObjectContainmentEList<Expression>(Expression.class, this, MiniJavaPackage.METHOD_CALL__ARGS);
		}
		return args;
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL__RECEIVER :
				return basicSetReceiver(null, msgs);
			case MiniJavaPackage.METHOD_CALL__ARGS :
				return ((InternalEList<?>) getArgs()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL__RECEIVER :
				return getReceiver();
			case MiniJavaPackage.METHOD_CALL__METHOD :
				if (resolve)
					return getMethod();
				return basicGetMethod();
			case MiniJavaPackage.METHOD_CALL__ARGS :
				return getArgs();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL__RECEIVER :
				setReceiver((Expression) newValue);
				return;
			case MiniJavaPackage.METHOD_CALL__METHOD :
				setMethod((Method) newValue);
				return;
			case MiniJavaPackage.METHOD_CALL__ARGS :
				getArgs().clear();
				getArgs().addAll((Collection<? extends Expression>) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL__RECEIVER :
				setReceiver((Expression) null);
				return;
			case MiniJavaPackage.METHOD_CALL__METHOD :
				setMethod((Method) null);
				return;
			case MiniJavaPackage.METHOD_CALL__ARGS :
				getArgs().clear();
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.METHOD_CALL__RECEIVER :
				return receiver != null;
			case MiniJavaPackage.METHOD_CALL__METHOD :
				return method != null;
			case MiniJavaPackage.METHOD_CALL__ARGS :
				return args != null && !args.isEmpty();
		}
		return super.eIsSet(featureID);
	}

	public Value evaluateExpression(State state) {
		Value result;
		ObjectRefValue realReceiver0 = ((ObjectRefValue) (((ObjectRefValue) (((Expression) (this.getReceiver())).evaluateExpression((State) (state))))));
		ObjectInstance realReceiver = ((ObjectInstance) (realReceiver0.getInstance()));
		Method realMethod = ((Method) (((Method) (((Method) (this.getMethod())).findOverride((Clazz) (realReceiver.getType()))))));
		Context newContext = ((Context) (MiniJavaFactory.eINSTANCE.createContext()));
		int argsLength = ((int) (CollectionService.size(this.getArgs())));
		int i = ((int) (0));
		while ((i) < (argsLength)) {
			Expression arg = ((Expression) (CollectionService.get(this.getArgs(), i)));
			Parameter param = ((Parameter) (CollectionService.get(realMethod.getParams(), i)));
			SymbolBinding binding = ((SymbolBinding) (MiniJavaFactory.eINSTANCE.createSymbolBinding()));
			binding.setSymbol(param);
			binding.setValue(((Expression) (arg)).evaluateExpression((State) (state)));
			newContext.getBindings().add(binding);
			i = (i) + (1);
		}
		MethodCall2 call = ((MethodCall2) (MiniJavaFactory.eINSTANCE.createMethodCall2()));
		call.setMethodcall(this);
		((State) (state)).pushNewFrame((ObjectInstance) (realReceiver), (MethodCall2) (call), (Context) (newContext));
		((MethodCall) (this)).call((Method) (realMethod), (State) (state));
		Value returnValue = ((Value) (((State) (state)).findCurrentFrame().getReturnValue()));
		((State) (state)).popCurrentFrame();
		result = (Value) (returnValue) ;
		return result;
	}

	public void call(Method realMethod, State state) {
		((Method) (realMethod)).call((State) (state));
	}
}
